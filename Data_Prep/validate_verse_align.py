"""
Does verse-level CTC actually recover word boundaries? Calibrate on R6.
=======================================================================
verse_align.py rejects a verse whose forced-alignment score falls below a
threshold. Picking that threshold by eye would be circular — it would keep
whatever looks tidy. R6 removes the guesswork: its boundaries are human
verified, so the same alignment can be scored against the truth.

This runs the verse-level alignment over PocketTorah audio and reports the word
boundary error against gold, bucketed by alignment score. That gives both the
threshold and, more importantly, the answer to whether the method works at all:
if verse-level CTC cannot reproduce gold boundaries on clean teaching audio, it
has no business being trusted on YouTube chanting.

Usage:
  python Data_Prep/validate_verse_align.py --recordings 6
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verse_align as VA  # noqa: E402
from ctc_align import load_recording, SR  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", type=int, default=6)
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] == "R6"]
    by_file = defaultdict(list)
    for r in records:
        by_file[r["audio_file"]].append(r)

    rows = []          # (score, |d_start| ms, |d_end| ms, d_start, d_end)
    for n, audio_file in enumerate(sorted(by_file)[: args.recordings], 1):
        path = PROJECT_DIR / audio_file
        if not path.exists():
            continue
        t0 = time.time()
        y_full = load_recording(path)
        by_verse = defaultdict(list)
        for r in by_file[audio_file]:
            by_verse[r["verse_ref"]].append(r)

        n_ok = 0
        for ref, items in by_verse.items():
            try:
                book, cv = ref.rsplit(" ", 1)
                chapter, verse = (int(x) for x in cv.split(":"))
            except ValueError:
                continue
            words = VA.verse_words(book, chapter, verse)
            if not words:
                continue
            items.sort(key=lambda z: z["start"])
            lo = max(0.0, items[0]["start"] - VA.MACRO_PAD)
            hi = min(len(y_full) / SR, items[-1]["end"] + VA.MACRO_PAD)
            if not (0.4 < hi - lo <= VA.MAX_WINDOW_SEC):
                continue

            spans, score, _ = VA.align_verse(y_full[int(lo * SR):int(hi * SR)], words)
            if spans is None:
                continue
            n_ok += 1

            import difflib
            from clean_align_v2 import skeleton
            a = [skeleton(r["word_clean"]) for r in items]
            b = [skeleton(w["norm"]) for w in words]
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                    a=a, b=b, autojunk=False).get_opcodes():
                if tag != "equal":
                    continue
                for k in range(i2 - i1):
                    rec, w_i = items[i1 + k], j1 + k
                    if w_i not in spans:
                        continue
                    ws, we = lo + spans[w_i][0], lo + spans[w_i][1]
                    ds = (ws - rec["start"]) * 1000
                    de = (we - rec["end"]) * 1000
                    rows.append((score, abs(ds), abs(de), ds, de))
        print(f"[{n}] {path.name}: {n_ok} verses aligned ({time.time() - t0:.0f}s)",
              flush=True)

    if not rows:
        print("nothing aligned")
        return

    arr = np.array(rows)
    scores = arr[:, 0]
    print(f"\nwords compared against gold: {len(rows):,}")
    print(f"alignment score: median {np.median(scores):.2f}  "
          f"p10 {np.percentile(scores, 10):.2f}  p90 {np.percentile(scores, 90):.2f}")

    print(f"\n{'score bucket':<18}{'n':>7}{'|start| p50':>13}{'p90':>8}"
          f"{'|end| p50':>11}{'p90':>8}")
    print("-" * 66)
    edges = [-np.inf, -3.0, -2.5, -2.0, -1.5, np.inf]
    for i in range(len(edges) - 1):
        m = (scores >= edges[i]) & (scores < edges[i + 1])
        if m.sum() < 20:
            continue
        lab = f"{edges[i]:.1f} .. {edges[i+1]:.1f}".replace("-inf", "  -inf") \
                                                    .replace("inf", " inf")
        print(f"{lab:<18}{m.sum():>7,}{np.median(arr[m, 1]):>12.0f}m"
              f"{np.percentile(arr[m, 1], 90):>7.0f}m"
              f"{np.median(arr[m, 2]):>10.0f}m{np.percentile(arr[m, 2], 90):>7.0f}m")
    print("-" * 66)

    ds, de = arr[:, 3], arr[:, 4]
    print(f"\noverall vs gold: start median {np.median(ds):+.0f} ms "
          f"(|d| p50 {np.median(np.abs(ds)):.0f}), "
          f"end median {np.median(de):+.0f} ms (|d| p50 {np.median(np.abs(de)):.0f})")
    print("\nFor reference, Whisper's own error against the same gold onsets was\n"
          "-200 ms median at the start and -284 ms at the end. Verse-level CTC is\n"
          "only worth adopting where it beats that clearly.")

    # CTC emits a character only once it has accumulated enough evidence, so its
    # onsets lag the acoustics by a roughly constant amount. That part of the
    # error is a calibration constant, not noise: estimate it on gold and remove
    # it. What remains is the error that actually limits the method.
    bs, be = np.median(ds), np.median(de)
    print(f"\nremoving the constant emission lag (start {bs:+.0f} ms, end {be:+.0f} ms):")
    print(f"{'':<10}{'|d| p50':>10}{'p75':>8}{'p90':>8}{'>200ms':>9}")
    print("-" * 45)
    for name, arr, bias in (("start", ds, bs), ("end", de, be)):
        c = np.abs(arr - bias)
        print(f"{name:<10}{np.median(c):>9.0f}m{np.percentile(c, 75):>7.0f}m"
              f"{np.percentile(c, 90):>7.0f}m{100 * np.mean(c > 200):>8.1f}%")
    print("-" * 45)
    print("Compare the corrected |d| p50 against Whisper's 214 / 288 ms. The gap\n"
          "between them is what re-aligning R1/R2 actually buys.")


if __name__ == "__main__":
    main()
