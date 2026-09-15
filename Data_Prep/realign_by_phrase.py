"""
Re-align R1/R2/R8 phrase-by-phrase instead of verse-by-verse.
==============================================================
THE HYPOTHESIS, FROM A DIRECT USER OBSERVATION
------------------------------------------------
Every word in this project's main corpus was produced by forcing an entire
VERSE onto its audio with CTC (verse_align.py): R1 averages 6.0 words per
verse, up to 20. Ben13's reading corpus, cut at the source into phrases
averaging 2.9 words, needed no such long-range alignment -- and scored 89.9%
alone vs 77.7% for R1+R2+R8 combined on the same 11 classes. The gap was
attributed to voice diversity, but the ALIGNMENT GRANULARITY was never
isolated as a variable. This script isolates it.

METHOD: two-pass alignment, no new audio needed
-------------------------------------------------
Pass 1 (already done, in aligned_dataset_merged.json): coarse per-verse CTC
gives approximate word boundaries -- enough to locate each word's
neighbourhood, not enough to trust the exact edges.

Pass 2 (this script): split each verse into PHRASES at the same disjunctive
accents Ben13's own segmentation uses (taam_grammar.py -- derived
independently of Ben13, confirmed to agree with it in Milestone-era checks).
For each phrase, take a small padded window around its pass-1 words and
re-run align_verse() on just that phrase's own word list. A 2-4 word
re-alignment has the same short-range advantage Ben13's corpus has
structurally, without recording anyone.

OUTPUT
------
features_v3_word/{reader}P_X.npy + _meta.json (P = phrase-realigned), in the
exact format 00_validate_and_prepare_data.py consumes -- a direct drop-in
comparison against the original {reader}_X.npy from the same audio.

Usage:
  python Data_Prep/realign_by_phrase.py --readers R1 --limit 300   # trial
  python Data_Prep/realign_by_phrase.py --readers R1 R2 R8
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import difflib                                         # noqa: E402
from clean_align_v2 import normalize_word, skeleton    # noqa: E402
from ctc_align import SR, load_recording              # noqa: E402
from verse_align import align_verse                   # noqa: E402
import taam_grammar as tg                              # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
CACHE = PROJECT_DIR / "Data_Prep" / "sefaria_cache"
OUT_DIR = PROJECT_DIR / "features_v3_word"
BOOK_NAMES = {"Genesis": "Genesis", "Exodus": "Exodus", "Leviticus": "Leviticus",
              "Numbers": "Numbers", "Deuteronomy": "Deuteronomy"}

N_MELS, FMAX, MAX_FRAMES, PAD_DB = 128, 8000, 87, -80.0
PHRASE_PAD = 0.35     # slack around a phrase's coarse-pass word span
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "5"))


def word_spectrogram(y, start, end):
    s = max(0, int((start - 0.05) * SR))
    e = min(len(y), int((end + 0.05) * SR))
    if e <= s or (e - s) < SR * 0.1:
        return None
    mel = librosa.feature.melspectrogram(y=y[s:e], sr=SR, n_mels=N_MELS, fmax=FMAX)
    db = librosa.power_to_db(mel, ref=np.max)
    if db.shape[1] < MAX_FRAMES:
        db = np.pad(db, ((0, 0), (0, MAX_FRAMES - db.shape[1])),
                    mode="constant", constant_values=PAD_DB)
    return db[:, :MAX_FRAMES].astype(np.float32)


def sefaria_verse(book, chapter, verse):
    p = CACHE / f"{book}.{chapter}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        vs = json.load(f)
    return vs[verse - 1] if 0 < verse <= len(vs) else None


def phrase_chunks(tokens):
    """Split a verse's (word, taam) tokens into phrases at each disjunctive
    accent -- the identical rule Ben13's own segments follow. Returns a list
    of (lo, hi) index ranges."""
    chunks, start = [], 0
    for i, (_, taam) in enumerate(tokens):
        if taam in tg.DISJUNCTIVE_RANK:
            chunks.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        chunks.append((start, len(tokens)))
    return chunks


def process_verse(args):
    """One (reader, audio_file, verse_ref, coarse_words) -> (specs, metas)."""
    reader, audio_file, verse_ref, coarse = args
    specs, metas = [], []
    try:
        book, cv = verse_ref.rsplit(" ", 1)
        chapter, verse_n = (int(x) for x in cv.split(":"))
    except ValueError:
        return specs, metas, "bad_ref"
    raw = sefaria_verse(book, chapter, verse_n)
    if raw is None:
        return specs, metas, "no_sefaria"
    tokens = tg.verse_tokens(raw)
    if not tokens:
        return specs, metas, "no_tokens"

    # `coarse` only holds the ~2/3 of the verse's words that survived
    # clean_align_v2's own matching in pass 1 (see verse_align.py's own
    # docstring) -- so it is a SUBSET of `tokens`, not the same length.
    # Map it back onto the full verse the same way verse_align.py does:
    # skeleton (bare-consonant) sequence matching, keep only the aligned
    # runs, and use the FULL-VERSE taam (from taam_grammar, independent of
    # clean_align_v2's own decoding) as ground truth for phrase boundaries.
    # skeleton() expects already-normalized bare-consonant text (that's what
    # word_clean already is); feeding it raw pointed Sefaria text left
    # niqqud/cantillation marks in b_sk and every comparison failed. Run the
    # same normalize_word() pass verse_align.py itself uses first.
    a_sk = [skeleton(c["word_clean"]) for c in coarse]
    b_sk = [skeleton(normalize_word(w)) for w, _ in tokens]
    opcodes = difflib.SequenceMatcher(a=a_sk, b=b_sk, autojunk=False).get_opcodes()
    coarse_by_tok = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            coarse_by_tok[j1 + k] = coarse[i1 + k]
    if not coarse_by_tok:
        # nothing in the verse survived pass 1 at all -- no anchor anywhere
        return specs, metas, "no_anchor"
    # No whole-verse coverage threshold beyond that: median survival from
    # pass 1 is ~38% (matches verse_align.py's own "about a third" note), so
    # requiring most of the verse would reject the majority of real verses
    # for no reason -- the actual requirement is an anchor per PHRASE, a
    # smaller unit, checked individually below.

    path = PROJECT_DIR / audio_file
    if not path.exists():
        return specs, metas, "no_audio"
    y_full = load_recording(path)

    for lo, hi in phrase_chunks(tokens):
        phrase_toks = tokens[lo:hi]
        words = [{"raw": w, "norm": normalize_word(w)} for w, _ in phrase_toks]
        c_span = [coarse_by_tok[i] for i in range(lo, hi) if i in coarse_by_tok]
        if not c_span:
            continue        # no surviving anchor in this phrase at all
        t0 = max(0.0, min(c["start"] for c in c_span) - PHRASE_PAD)
        t1 = min(len(y_full) / SR, max(c["end"] for c in c_span) + PHRASE_PAD)
        if t1 - t0 < 0.1 or t1 - t0 > 20:
            continue
        y = y_full[int(t0 * SR):int(t1 * SR)]
        spans, score, _ = align_verse(y, words)
        if spans is None:
            continue
        for wi, (_, taam) in enumerate(phrase_toks):
            if taam is None or wi not in spans:
                continue
            st, en = spans[wi]
            spec = word_spectrogram(y, st, en)
            if spec is None:
                continue
            specs.append(spec)
            metas.append({
                "taam": taam, "word_clean": words[wi]["norm"],
                "verse_ref": verse_ref, "reader": reader,
                "start": round(t0 + st, 3), "end": round(t0 + en, 3),
                "window": "word", "align": "phrase_realign",
                "phrase_words": hi - lo,
                "ctc_score": round(float(score), 3) if score is not None else None,
            })
    return specs, metas, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["R1", "R2", "R8"])
    ap.add_argument("--limit", type=int, default=0, help="verses per reader, for a trial")
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] in args.readers]

    for reader in args.readers:
        by_verse = defaultdict(list)
        for r in records:
            if r["reader"] != reader:
                continue
            by_verse[(r["audio_file"], r["verse_ref"])].append(r)
        for v in by_verse.values():
            v.sort(key=lambda r: r["start"])

        jobs = [(reader, af, vr, ws) for (af, vr), ws in by_verse.items()]
        if args.limit:
            jobs = jobs[: args.limit]
        print(f"\n{reader}: {len(jobs):,} verses to re-align "
              f"({WORKERS} workers)", flush=True)

        specs, meta, stats = [], [], defaultdict(int)
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(process_verse, j) for j in jobs]
            for n, fut in enumerate(as_completed(futures), 1):
                s, m, status = fut.result()
                specs.extend(s)
                meta.extend(m)
                stats[status] += 1
                if n % 1000 == 0:
                    print(f"  {n}/{len(jobs)}  words={len(meta):,}  {dict(stats)}",
                          flush=True)

        print(f"{reader}: {dict(stats)}")
        if not specs:
            print(f"{reader}: nothing extracted, skipping save")
            continue
        arr = np.stack(specs)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(OUT_DIR / f"{reader}P_X.npy", arr)
        with open(OUT_DIR / f"{reader}P_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"{reader}P: {arr.shape} -> features_v3_word/{reader}P_X.npy")


if __name__ == "__main__":
    main()
