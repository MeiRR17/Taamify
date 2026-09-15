"""
Word-level features from Ben13's reading corpus.
================================================
WHY THIS IS DIFFERENT FROM EVERY EARLIER ALIGNMENT IN THIS PROJECT
------------------------------------------------------------------
Every previous corpus here was aligned by forcing a whole verse -- sometimes
a whole 40-minute parasha -- onto its text with CTC. That is where the
project's data bugs came from: a long span gives the aligner room to drift,
and when the text is even slightly wrong the error cascades silently through
every following word.

Ben13's reading corpus removes most of that risk by construction. It is
already cut into phrases at the disjunctive accents, averaging **2.9 words
per segment**. Aligning three words against three known words is a different
problem from aligning forty. The phrase boundary is given, not inferred, so
drift cannot accumulate across it.

WHAT THIS PRODUCES
------------------
Per-word mel-spectrograms in exactly the format 00_validate_and_prepare_data.py
already consumes (features_v3_word/<reader>_X.npy + _meta.json, 128x87), so
this arrives as a new reader -- BEN13R -- and every existing script, split,
and training run works on it unchanged.

Scale: 27,788 Torah segments hold 68,609 words. The best existing reader (R1)
contributes 27,031. So this one source is roughly 2.5x larger than anything in
the corpus today, and covers the entire Torah rather than 91% of it.

QUALITY IS RECORDED, NOT ASSUMED
--------------------------------
Every word keeps the CTC score of the segment it came from, and segments whose
alignment fails or scores badly are flagged rather than silently kept. The
whole point of this corpus is that its provenance is checkable; that only
holds if the numbers travel with the data.

Usage:
  python Data_Prep/extract_ben13_reading_words.py --limit 200   # trial
  python Data_Prep/extract_ben13_reading_words.py --books 1 2 3 4 5
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import normalize_word          # noqa: E402
from ctc_align import SR, load_recording           # noqa: E402
from verse_align import align_verse                # noqa: E402
import taam_grammar as tg                          # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "ben13_reading" / "manifest_yerushalmi.json"
MANIFEST = DEFAULT_MANIFEST   # overridden by --manifest
OUT_DIR = PROJECT_DIR / "features_v3_word"
READER = "BEN13R"

N_MELS, FMAX, MAX_FRAMES, PAD_DB = 128, 8000, 87, -80.0
FLAG_SCORE = -3.0          # segments below this are recorded as low-confidence

# 27,788 segments, each needing an audio decode, a CTC pass and a mel
# transform. Serially that is hours of leaving 9 of 10 cores idle on a
# machine with 64GB to spare. Processes rather than threads because the work
# is CPU-bound and each worker loads its own copy of the CTC model.
#
# 8 workers deadlocked the first full run at 22,000/27,693: each worker holds
# its own ~1.2GB CTC model while the parent accumulates every spectrogram
# (68k x 128 x 87 floats is ~2.4GB), and with swap already near full a worker
# was killed, leaving as_completed() waiting forever on a broken pool.
# Fixed on both sides -- fewer workers, and results are flushed to shards
# instead of being held in the parent.
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "5"))
CHUNK = 2000          # segments per shard; bounds parent memory


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


def process_segment(r):
    """One segment -> (specs, metas, stat-deltas). Runs in a worker process."""
    st = {"segments": 0, "aligned": 0, "align_failed": 0, "no_audio": 0,
          "words": 0, "low_conf": 0, "spec_failed": 0}
    specs, metas = [], []
    path = PROJECT_DIR / r["local_path"]
    if not path.exists():
        st["no_audio"] += 1
        return specs, metas, st
    st["segments"] += 1

    toks = tg.verse_tokens(r["text"])
    words = [{"raw": w, "norm": normalize_word(w)} for w, _ in toks]
    taams = [t for _, t in toks]
    if not words:
        return specs, metas, st
    try:
        y = load_recording(path)
    except Exception:
        st["no_audio"] += 1
        return specs, metas, st

    if len(words) == 1:
        # a one-word segment needs no alignment at all -- the phrase boundary
        # IS the word boundary, which is this corpus's whole advantage
        spans, score = {0: (0.0, len(y) / SR)}, 0.0
    else:
        spans, score, _ = align_verse(y, words)
        if spans is None:
            st["align_failed"] += 1
            return specs, metas, st
    st["aligned"] += 1
    low = score is not None and score < FLAG_SCORE
    if low:
        st["low_conf"] += 1

    for wi, taam in enumerate(taams):
        if taam is None or wi not in spans:
            continue
        start, end = spans[wi]
        spec = word_spectrogram(y, start, end)
        if spec is None:
            st["spec_failed"] += 1
            continue
        specs.append(spec)
        metas.append({
            "taam": taam, "word_clean": words[wi]["norm"],
            "word_raw": words[wi]["raw"], "parasha": r["parasha"],
            "verse_ref": f"{r['book']}.{r['perek']}.{r['pasuk']}",
            "start": round(float(start), 3), "end": round(float(end), 3),
            "window": "word", "align": "ben13_segment_ctc",
            "segment_block": r["block"],
            "ctc_score": round(float(score), 3) if score is not None else None,
            "low_confidence": bool(low),
        })
        st["words"] += 1
    return specs, metas, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=["1", "2", "3", "4", "5"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-reader", default=READER)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="ben13 manifest to read; ben13.co.il serves three\n"
                         "nusachim (yerushalmi/ashkenazi/maroko) and each is a\n"
                         "separate clean voice with its own manifest")
    args = ap.parse_args()

    global MANIFEST
    MANIFEST = Path(args.manifest)
    with open(MANIFEST, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if r["book"] in args.books]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows):,} segments from books {args.books}")

    X, meta = [], []
    stats = {"segments": 0, "aligned": 0, "align_failed": 0, "no_audio": 0,
             "words": 0, "low_conf": 0, "spec_failed": 0}

    print(f"extracting with {WORKERS} workers, {CHUNK}-segment shards")
    shard_dir = OUT_DIR / f".{args.out_reader}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths, done = [], 0

    # One pool per chunk: a pool that breaks takes only its own chunk with
    # it, and the shard already written stays valid. The previous single
    # long-lived pool lost the entire run when one worker died.
    for ci in range(0, len(rows), CHUNK):
        chunk = rows[ci:ci + CHUNK]
        sp = shard_dir / f"{ci:06d}.npz"
        mp_ = shard_dir / f"{ci:06d}.json"
        if sp.exists() and mp_.exists():
            shard_paths.append((sp, mp_))
            done += len(chunk)
            print(f"  shard {ci:06d} cached", flush=True)
            continue
        cX, cM = [], []
        try:
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                for specs, metas, st in pool.map(process_segment, chunk, chunksize=8):
                    for k, v in st.items():
                        stats[k] += v
                    cX.extend(specs)
                    cM.extend(metas)
        except Exception as e:
            print(f"  chunk {ci} failed ({e}) — continuing", flush=True)
        if cX:
            np.savez_compressed(sp, X=np.stack(cX))
            with open(mp_, "w", encoding="utf-8") as f:
                json.dump(cM, f, ensure_ascii=False)
            shard_paths.append((sp, mp_))
        done += len(chunk)
        print(f"  {done}/{len(rows)}  words={stats['words']:,} "
              f"failed={stats['align_failed']}", flush=True)

    for sp, mp_ in shard_paths:
        X.append(np.load(sp)["X"])
        with open(mp_, encoding="utf-8") as f:
            meta.extend(json.load(f))
    X = [a for a in X]


    if not X:
        print("nothing extracted")
        return
    arr = np.concatenate(X) if X[0].ndim == 3 else np.stack(X)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{args.out_reader}_X.npy", arr)
    with open(OUT_DIR / f"{args.out_reader}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    for k, v in stats.items():
        print(f"  {k:<16}{v:,}")
    print(f"  shape {arr.shape} -> {OUT_DIR.name}/{args.out_reader}_X.npy")
    from collections import Counter
    c = Counter(m["taam"] for m in meta)
    print(f"\n  {len(c)} classes:")
    for t, n in c.most_common():
        print(f"    {tg.TAAM_HE.get(t, t):<16}{n:>7,}")


if __name__ == "__main__":
    main()
