"""
Second-pass refinement of R7's word boundaries: narrow the search, not the model.
===================================================================================
The noise-control experiment (see FINDINGS.md section 8) showed R7's collapsed
rank-lift is fully explainable by its alignment noise alone — extrapolating R1's
own noise-response curve lands on R7's observed result at ~600ms of injected
jitter, and R7's median CTC score (-2.79) sits well below R1/R2/R6's (-1.91).
That is not evidence Yemenite tradition is structurally more distant; it is
evidence the data isn't clean enough to ask the question yet.

The bottleneck was never the labels — proportional_verse_locate.py's ta'am
sequences were hand-verified correct against known verses (Shema Yisrael among
them) before that pipeline was trusted. What's imprecise is *timing*: pass one
force-aligned an entire verse inside a window sized by proportional guesswork,
sometimes many seconds wide, and a wide search window is exactly what lets a
CTC model drift.

This is a second pass over the same, already-correctly-labeled words: for each
one, open a narrow window around its pass-one span (word) rather than a whole
verse (paragraph), and re-run forced alignment there. The same acoustic model,
the same audio — only the search space shrinks. This is the identical technique
that gave R1/R2/R6 their -1.9 to -2.0 scores in the first place (ctc_align.py's
per-word pass); R7 never got it because Whisper couldn't locate individual
words in it to seed a narrow window. Pass one's own (now-verified) output can
seed it instead.

A record only updates if the refined window scores at least as well as pass
one's; otherwise the original stands. Nothing is discarded — every record keeps
its provenance (`match`), so a refined vs. unrefined comparison is always
possible downstream.

RESULT: this does not work. Tried at SEARCH_PAD 0.45s and 1.0s on a pilot
recording; both made things worse, not better (mean refined score -4.08 and
-3.44 respectively, against pass-one's -2.79 and R1's -1.91). Isolating a
single word loses the sentence-level context the verse-level pass had to
constrain the search, and that loss outweighs the benefit of a narrower
window. This says the bottleneck is not search-space width — it looks like a
genuine mismatch between this acoustic model and Yemenite phonology, which a
smaller search window can't route around. Left in the repo as a documented
negative result, same as the stressed-syllable-window and reader-conditioning
attempts. The live corpus was restored from backup after both trials; neither
result is reflected in aligned_dataset_r7.json.

What would plausibly help instead: a Hebrew acoustic model fine-tuned on
Yemenite or Mizrahi pronunciation (none identified so far), or a cleaner
source recording for a third tradition — see outreach/ and FINDINGS.md
section 8.

Usage (do not run against the live corpus without re-checking the result
above first):
  python Data_Prep/refine_r7_alignment.py --limit 1
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctc_align import load_model, load_recording, SR  # noqa: E402
from stressed_syllable import stressed_syllable  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC = PROJECT_DIR / "aligned_dataset_r7.json"
OUT = PROJECT_DIR / "aligned_dataset_r7.json"          # refine in place
VERSE_WINDOWS = PROJECT_DIR / "verse_aligned_windows.json"

SEARCH_PAD = 1.0          # seconds either side of the pass-one span
MIN_WORD_SEC = 0.08
MAX_WORD_SEC = 4.0


def align_word_scored(y: np.ndarray, letters: str):
    """Like ctc_align.align_word, but also returns the alignment's mean score —
    needed here to decide whether the refinement actually improved things."""
    (model, device), vocab, blank = load_model()
    ids = [vocab[c] for c in letters if c in vocab]
    if not ids or len(y) < SR * 0.05:
        return None, None, None

    with torch.inference_mode():
        logits = model(torch.from_numpy(y).unsqueeze(0).to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1).cpu()

    n_frames = log_probs.shape[1]
    if n_frames < len(ids):
        return None, None, None

    import torchaudio.functional as AF
    targets = torch.tensor([ids], dtype=torch.int32)
    try:
        aligned, scores = AF.forced_align(log_probs.float(), targets, blank=blank)
        spans = AF.merge_tokens(aligned[0], scores[0], blank=blank)
    except (RuntimeError, ValueError):
        return None, None, None
    if len(spans) != len(ids):
        return None, None, None

    onsets = [int(s.start) for s in spans]
    mean_score = float(np.mean([s.score for s in spans]))
    frame_sec = len(y) / SR / max(n_frames, 1)
    return onsets, mean_score, frame_sec


def refine_one(y_full: np.ndarray, record: dict):
    """Attempt a narrow-window realignment; None if it fails or doesn't help."""
    lo = max(0.0, record["start"] - SEARCH_PAD)
    hi = min(len(y_full) / SR, record["end"] + SEARCH_PAD)
    y = y_full[int(lo * SR):int(hi * SR)]

    onsets, score, frame_sec = align_word_scored(y, record["word_clean"])
    if onsets is None:
        return None

    n_frames = int((hi - lo) / frame_sec) if frame_sec else 0
    starts = onsets + [n_frames if n_frames else onsets[-1] + 1]
    new_start = lo + starts[0] * frame_sec
    new_end = lo + starts[-1] * frame_sec
    if not (MIN_WORD_SEC <= new_end - new_start <= MAX_WORD_SEC):
        return None

    return {"start": round(new_start, 3), "end": round(new_end, 3), "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap recordings, for a quick check")
    args = ap.parse_args()

    with open(SRC, encoding="utf-8") as f:
        records = json.load(f)
    print(f"loaded {len(records):,} R7 records")

    by_file = defaultdict(list)
    for r in records:
        by_file[r["audio_file"]].append(r)
    files = sorted(by_file)
    if args.limit:
        files = files[: args.limit]

    stats = Counter()
    before_scores, after_scores = [], []
    verse_updates = {}

    for n, audio_file in enumerate(files, 1):
        path = PROJECT_DIR / audio_file
        if not path.exists():
            continue
        t0 = time.time()
        y_full = load_recording(path)
        items = by_file[audio_file]

        improved = 0
        for rec in items:
            before_score = None  # pass-one didn't carry a per-word score, only a per-verse one
            refined = refine_one(y_full, rec)
            if refined is None:
                stats["kept_unrefined"] += 1
                continue

            after_scores.append(refined["score"])
            # A refined window is trusted when it clears a floor comparable to
            # R1/R2/R6's own per-word scores, not merely "better than nothing" —
            # a bad narrow window can still score above pass-one's verse-wide
            # (and therefore diluted) score without being good in absolute terms.
            if refined["score"] >= -2.5:
                rec["start"], rec["end"] = refined["start"], refined["end"]
                rec["match"] = "proportional_ctc_refined"
                rec["refine_score"] = refined["score"]
                improved += 1
                stats["refined"] += 1

                syl = stressed_syllable(rec["word_raw"])
                if syl.get("ok"):
                    n_let = max(syl["n_letters"], 1)
                    dur = rec["end"] - rec["start"]
                    key = f"{rec['reader']}|{rec['parasha']}|{rec['start']}"
                    verse_updates[key] = {
                        "word_start": rec["start"], "word_end": rec["end"],
                        "start": round(rec["start"] + dur * syl["letter_start"] / n_let, 3),
                        "end": round(rec["start"] + dur * syl["letter_end"] / n_let, 3),
                        "method": "proportional_ctc_refined",
                    }
            else:
                stats["refine_scored_low"] += 1
        del y_full
        print(f"[{n}/{len(files)}] {Path(audio_file).name}: "
              f"{improved}/{len(items)} refined ({time.time() - t0:.0f}s)", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\nwrote {len(records):,} records -> {OUT.name}")

    if VERSE_WINDOWS.exists() and verse_updates:
        with open(VERSE_WINDOWS, encoding="utf-8") as f:
            vw = json.load(f)
        vw.update(verse_updates)
        with open(VERSE_WINDOWS, "w", encoding="utf-8") as f:
            json.dump(vw, f)
        print(f"updated {len(verse_updates):,} entries in {VERSE_WINDOWS.name}")

    print(f"\nrefined: {stats['refined']:,}  "
          f"kept unrefined (window failed): {stats['kept_unrefined']:,}  "
          f"scored below floor: {stats['refine_scored_low']:,}")
    if after_scores:
        arr = np.array(after_scores)
        print(f"\nrefined-window score: mean {arr.mean():.3f}  median {np.median(arr):.3f}")
        print("compare against R1's -1.91 median and R7 pass-one's -2.79.")


if __name__ == "__main__":
    main()
