"""
Rebuild word boundaries from the acoustics, one verse at a time.
================================================================
Measured against PocketTorah's human-verified onsets, Whisper places a word
boundary a median 200 ms early at the start and 284 ms early at the end, with
53% of starts off by more than 200 ms and a p90 of 570 ms. A stressed syllable
is roughly 460 ms. So the Whisper windows behind R1/R2 — 43% of the corpus —
open on the tail of the previous word and close mid-syllable, exactly what the
clips sound like.

A constant shift will not repair that: the bias is systematic but the spread is
not. The boundaries have to come from the audio.

Whisper is kept only as a macro-locator: its word times, loose as they are,
still say roughly where a verse sits inside a 40-minute recording. Everything
finer is re-derived by forcing the *canonical verse* onto that window with CTC.

The target is the whole verse from Sefaria, not the surviving records.
clean_align_v2 dropped about a third of the words; aligning only the survivors
would ask CTC to stretch them across audio that also contains the dropped ones,
producing a confident, wrong alignment. Words are separated by the model's word
delimiter so boundaries are a token the model predicts rather than a gap we
guess at.

Because a forced alignment is monotonic and spans the whole verse, no word can
end early without pushing its neighbour — which is precisely the failure mode
the per-word windows had.

Usage:
  python Data_Prep/verse_align.py --readers R1 R2 --limit 40   # pilot
  python Data_Prep/verse_align.py --readers R1 R2 --all
"""

import argparse
import difflib
import html
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import (  # noqa: E402
    CACHE_DIR, CURLY_MARKERS, HTML_TAGS, MAQAF, normalize_word, skeleton,
)
from stressed_syllable import stressed_syllable  # noqa: E402
from ctc_align import load_model, SR  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
OUT = PROJECT_DIR / "verse_aligned_windows.json"
PARTS = Path.home() / ".cache" / "taamify" / "verse_parts"

MACRO_PAD = 0.6           # slack around Whisper's guess at the verse extent
MAX_WINDOW_SEC = 45.0     # a verse longer than this is a Whisper failure
MIN_SCORE = -3.2          # mean per-frame log-prob; calibrated, see below

# CTC emits a character only after accumulating enough evidence, so its onsets
# lag the acoustics by a near-constant amount. Measured against R6's gold onsets
# in validate_verse_align.py: +170 ms at word start, -63 ms at word end.
# Removing it takes the median boundary error from 186/170 ms to 114/161 ms,
# against Whisper's 214/288 ms on the same words. These are calibration
# constants for *this* model on *this* material; re-measure if either changes.
LAG_START = 0.170
LAG_END = -0.063

# Scores run about -2.2 median on chanted Hebrew because the model was trained
# on modern speech, so a threshold tuned for speech would reject everything.
# Accuracy does track the score (|start| p50 falls 211 -> 189 -> 119 ms across
# buckets), so the score is kept on every window as a quality weight rather than
# used as a hard gate that would throw away most of the corpus.
#
# The verse must be *spanned*, not densely covered: clean_align_v2 kept only
# ~41% of R1's words, but as long as the survivors reach both ends of the verse
# the macro window still contains it.
MAX_EDGE_GAP = 3          # verse words allowed to be missing at either end


def verse_words(book: str, chapter: int, verse: int) -> list:
    """Every pointed token of the verse, tokenized as clean_align_v2 does."""
    cache_file = CACHE_DIR / f"{book}.{chapter}.json"
    if not cache_file.exists():
        return []
    with open(cache_file, encoding="utf-8") as f:
        verses = json.load(f)
    if verse > len(verses):
        return []
    raw = CURLY_MARKERS.sub(" ", HTML_TAGS.sub("", html.unescape(verses[verse - 1])))
    raw = raw.replace(MAQAF, " ")
    out = []
    for tok in raw.split():
        norm = normalize_word(tok)
        if len(norm) >= 2:
            out.append({"raw": tok, "norm": norm})
    return out


def align_verse(y: np.ndarray, words: list):
    """Force the verse onto the audio; return per-word (start, end) in seconds.

    Returns (spans, mean_score, frame_sec) or (None, None, None).
    """
    (model, device), vocab, blank = load_model()
    delim = vocab.get("|")

    ids, owner = [], []           # owner[i] = index of the word token i belongs to
    for w_i, w in enumerate(words):
        if ids and delim is not None:
            ids.append(delim)
            owner.append(-1)      # separators belong to no word
        for ch in w["norm"]:
            if ch in vocab:
                ids.append(vocab[ch])
                owner.append(w_i)
    if not ids:
        return None, None, None

    with torch.inference_mode():
        logits = model(torch.from_numpy(y).unsqueeze(0).to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1).cpu().float()

    n_frames = log_probs.shape[1]
    if n_frames < len(ids):
        return None, None, None

    import torchaudio.functional as AF
    try:
        aligned, scores = AF.forced_align(
            log_probs, torch.tensor([ids], dtype=torch.int32), blank=blank)
        token_spans = AF.merge_tokens(aligned[0], scores[0], blank=blank)
    except (RuntimeError, ValueError):
        return None, None, None
    if len(token_spans) != len(ids):
        return None, None, None

    frame_sec = len(y) / SR / max(n_frames, 1)
    mean_score = float(np.mean([s.score for s in token_spans]))

    # A word runs from its first letter's onset to the onset of whatever comes
    # next — the following separator, or the next letter. Using the letter's own
    # emission run would measure CTC's spike, not the sound.
    spans = {}
    for i, w_i in enumerate(owner):
        if w_i < 0:
            continue
        start = token_spans[i].start
        end = token_spans[i + 1].start if i + 1 < len(token_spans) else n_frames
        if w_i not in spans:
            spans[w_i] = [start, end]
        else:
            spans[w_i][1] = end
    return ({k: (v[0] * frame_sec, v[1] * frame_sec) for k, v in spans.items()},
            mean_score, frame_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["R1", "R2"])
    ap.add_argument("--limit", type=int, default=40, help="recordings to process")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="print Whisper vs CTC boundary shifts")
    args = ap.parse_args()

    from ctc_align import load_recording

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] in args.readers]

    by_file = defaultdict(list)
    for r in records:
        by_file[r["audio_file"]].append(r)
    files = sorted(by_file)
    if not args.all:
        files = files[: args.limit]

    PARTS.mkdir(parents=True, exist_ok=True)
    written, stats = {}, Counter()
    shifts_start, shifts_end = [], []

    for n, audio_file in enumerate(files, 1):
        part = PARTS / (audio_file.replace("/", "__") + ".json")
        if part.exists():
            try:
                with open(part, encoding="utf-8") as f:
                    cached = json.load(f)
                written.update(cached["windows"])
                stats.update(cached["stats"])
                shifts_start.extend(cached.get("ds", []))
                shifts_end.extend(cached.get("de", []))
                print(f"[{n}/{len(files)}] {Path(audio_file).name} (cached)", flush=True)
                continue
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        path = PROJECT_DIR / audio_file
        if not path.exists():
            stats["missing_audio"] += 1
            continue
        t0 = time.time()
        y_full = load_recording(path)

        by_verse = defaultdict(list)
        for r in by_file[audio_file]:
            by_verse[r["verse_ref"]].append(r)

        part_windows, part_stats = {}, Counter()
        part_ds, part_de = [], []

        for ref, items in by_verse.items():
            try:
                book, cv = ref.rsplit(" ", 1)
                chapter, verse = (int(x) for x in cv.split(":"))
            except ValueError:
                part_stats["bad_ref"] += len(items)
                continue
            words = verse_words(book, chapter, verse)
            if not words:
                part_stats["no_verse_text"] += len(items)
                continue
            items.sort(key=lambda z: z["start"])
            # Does the surviving text reach both ends of the verse? If the first
            # or last words were dropped, the macro window starts or stops
            # inside the verse and CTC would squeeze the whole text into it.
            a_sk = [skeleton(r["word_clean"]) for r in items]
            b_sk = [skeleton(w["norm"]) for w in words]
            eq = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2
                  in difflib.SequenceMatcher(a=a_sk, b=b_sk,
                                             autojunk=False).get_opcodes()
                  if tag == "equal"]
            if not eq:
                part_stats["no_anchor"] += len(items)
                continue
            if eq[0][2] > MAX_EDGE_GAP or (len(words) - eq[-1][3]) > MAX_EDGE_GAP:
                part_stats["verse_not_spanned"] += len(items)
                continue

            lo = max(0.0, items[0]["start"] - MACRO_PAD)
            hi = min(len(y_full) / SR, items[-1]["end"] + MACRO_PAD)
            if hi - lo > MAX_WINDOW_SEC or hi - lo < 0.4:
                part_stats["bad_macro_window"] += len(items)
                continue

            y = y_full[int(lo * SR):int(hi * SR)]
            spans, score, _ = align_verse(y, words)
            if spans is None:
                part_stats["align_failed"] += len(items)
                continue
            if score < MIN_SCORE:
                part_stats["low_score"] += len(items)
                continue

            # Map the surviving records onto the canonical token sequence.
            a = a_sk
            b = b_sk
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                    a=a, b=b, autojunk=False).get_opcodes():
                if tag != "equal":
                    part_stats["unmatched"] += (i2 - i1)
                    continue
                for k in range(i2 - i1):
                    rec, w_i = items[i1 + k], j1 + k
                    if w_i not in spans:
                        continue
                    w_start = lo + spans[w_i][0] - LAG_START
                    w_end = lo + spans[w_i][1] - LAG_END
                    if w_end - w_start < 0.08:
                        part_stats["degenerate_word"] += 1
                        continue

                    syl = stressed_syllable(words[w_i]["raw"])
                    key = f"{rec['reader']}|{rec['parasha']}|{rec['start']}"
                    entry = {"word_start": round(w_start, 3),
                             "word_end": round(w_end, 3),
                             "score": round(score, 3)}
                    if syl.get("ok"):
                        n_let = max(syl["n_letters"], 1)
                        f0 = syl["letter_start"] / n_let
                        f1 = syl["letter_end"] / n_let
                        dur = w_end - w_start
                        entry["start"] = round(w_start + dur * f0, 3)
                        entry["end"] = round(w_start + dur * f1, 3)
                        entry["rule"] = syl["rule"]
                    part_windows[key] = entry
                    part_ds.append((rec["start"] - w_start) * 1000)
                    part_de.append((rec["end"] - w_end) * 1000)
                    part_stats["aligned"] += 1

        with open(part, "w", encoding="utf-8") as f:
            json.dump({"windows": part_windows, "stats": dict(part_stats),
                       "ds": part_ds, "de": part_de}, f)
        written.update(part_windows)
        stats.update(part_stats)
        shifts_start.extend(part_ds)
        shifts_end.extend(part_de)
        print(f"[{n}/{len(files)}] {Path(audio_file).name}: "
              f"{part_stats['aligned']} aligned ({time.time() - t0:.0f}s)", flush=True)

    print(f"\naligned {stats['aligned']:,} words")
    for k, v in stats.most_common():
        if k != "aligned":
            print(f"  {k:<20}{v:>8,}")

    if shifts_start:
        ds, de = np.array(shifts_start), np.array(shifts_end)
        print(f"\nhow far Whisper was from the acoustic boundary (ms):")
        print(f"{'':<8}{'median':>10}{'|d| p50':>10}{'p90':>9}{'>200ms':>9}")
        print("-" * 47)
        for name, arr in (("start", ds), ("end", de)):
            print(f"{name:<8}{np.median(arr):>9.0f}m{np.median(np.abs(arr)):>9.0f}m"
                  f"{np.percentile(np.abs(arr), 90):>8.0f}m"
                  f"{100 * np.mean(np.abs(arr) > 200):>8.1f}%")
        print("-" * 47)
        print("This is Whisper minus the lag-corrected CTC boundary — a disagreement,\n"
              "not an error: R1/R2 have no gold to score against. The lag constants\n"
              "were fitted on R6, whose Whisper bias (-200 ms) is not reproduced here,\n"
              "so they are an assumption for this material, not a measurement. What is\n"
              "established is that on R6, where truth exists, this method halves the\n"
              "boundary error; what is assumed is that it carries over.")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(written, f)
    print(f"\nwrote {len(written):,} windows -> {OUT.name}")


if __name__ == "__main__":
    main()
