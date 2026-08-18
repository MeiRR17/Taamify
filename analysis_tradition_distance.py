"""
Is the tradition grouping real, or did I just read the video titles?
====================================================================
Every claim in this round rests on R1 and R2 belonging to one musical tradition
and R6 to another. That grouping came from the YouTube titles — "נוסח ספרדי
ירושלמי" — and from PocketTorah being an American Ashkenazi recording. It is a
label, not a measurement, and if R1 and R2 are in fact as different from each
other as either is from R6, the whole framing collapses.

This measures it directly. For each ta'am and each reader, the pitch contour of
every instance is normalized (semitones relative to that reader's median f0,
time-warped to a fixed length) and averaged. The distance between two readers is
then the mean distance between their average contours, over the ta'amim they
share. Pitch is used rather than spectrum because timbre is exactly what we do
*not* want to compare: two readers can have very different voices and sing the
same tune.

The floor is set by splitting a single reader's own instances into two random
halves and measuring the distance between them. Anything above that floor is a
real difference; anything at it is sampling noise.

Prediction if the grouping is real:
    d(R1,R2)  ≈  within-reader floor  <<  d(R1,R6) ≈ d(R2,R6)

Usage:
  python analysis_tradition_distance.py --per-taam 60
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
VERSE_WINDOWS = PROJECT_DIR / "verse_aligned_windows.json"

SR = 22050
FMIN, FMAX = 65.0, 600.0
N_POINTS = 24          # contour resampled to this many points
MIN_VOICED = 8


def contour(y, sr=SR):
    """Pitch contour in semitones relative to the clip median, on a fixed grid."""
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr,
                                 frame_length=1024, hop_length=256)
    if f0 is None:
        return None
    ok = np.isfinite(f0) & voiced
    if ok.sum() < MIN_VOICED:
        return None
    ref = np.nanmedian(f0[ok])
    if not np.isfinite(ref) or ref <= 0:
        return None
    st = 12.0 * np.log2(f0[ok] / ref)
    # resample the voiced trajectory onto a common grid so words of different
    # length can be averaged; the shape is what carries the ta'am, not the pace
    x = np.linspace(0, 1, st.size)
    return np.interp(np.linspace(0, 1, N_POINTS), x, st)


def collect(records, verse_windows, per_taam, seed=0):
    """{reader: {taam: [contour, ...]}} sampled evenly across recordings."""
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: defaultdict(list))
    wanted = defaultdict(lambda: defaultdict(list))
    for r in records:
        wanted[r["reader"]][r["taam"]].append(r)

    out = defaultdict(lambda: defaultdict(list))
    for reader, taams in wanted.items():
        # group the picks by file so each recording is decoded once
        picks = []
        for taam, rs in taams.items():
            if len(rs) < per_taam:
                continue
            idx = rng.choice(len(rs), size=per_taam, replace=False)
            picks.extend(rs[i] for i in idx)
        by_file = defaultdict(list)
        for r in picks:
            by_file[r["audio_file"]].append(r)
        print(f"  {reader}: {len(picks)} words across {len(by_file)} recordings",
              flush=True)
        for n, (audio_file, items) in enumerate(sorted(by_file.items()), 1):
            path = PROJECT_DIR / audio_file
            if not path.exists():
                continue
            y_full, _ = librosa.load(str(path), sr=SR, mono=True)
            for r in items:
                key = f"{r['reader']}|{r['parasha']}|{r['start']}"
                va = verse_windows.get(key)
                s, e = ((va["word_start"], va["word_end"]) if va
                        else (r["start"], r["end"]))
                seg = y_full[int(s * SR):int(e * SR)]
                if len(seg) < SR * 0.15:
                    continue
                c = contour(seg)
                if c is not None:
                    out[reader][r["taam"]].append(c)
            del y_full
            if n % 25 == 0:
                print(f"    {reader}: {n}/{len(by_file)} files", flush=True)
    return out


def mean_contours(store, min_n=15):
    return {rd: {t: np.mean(v, axis=0) for t, v in d.items() if len(v) >= min_n}
            for rd, d in store.items()}


def reader_distance(a: dict, b: dict):
    shared = sorted(set(a) & set(b))
    if not shared:
        return None, 0
    d = [float(np.linalg.norm(a[t] - b[t]) / np.sqrt(N_POINTS)) for t in shared]
    return float(np.mean(d)), len(shared)


def within_reader_floor(store, reader, min_n=30, seed=0):
    """Split each ta'am's instances in half and compare the halves."""
    rng = np.random.default_rng(seed)
    h1, h2 = {}, {}
    for t, v in store[reader].items():
        if len(v) < min_n:
            continue
        arr = np.array(v)
        perm = rng.permutation(len(arr))
        half = len(arr) // 2
        h1[t] = arr[perm[:half]].mean(0)
        h2[t] = arr[perm[half:2 * half]].mean(0)
    return reader_distance(h1, h2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-taam", type=int, default=60,
                    help="instances sampled per reader per ta'am")
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        records = json.load(f)
    verse_windows = {}
    if VERSE_WINDOWS.exists():
        with open(VERSE_WINDOWS, encoding="utf-8") as f:
            verse_windows = json.load(f)

    print("collecting pitch contours:")
    store = collect(records, verse_windows, args.per_taam)
    means = mean_contours(store)
    for rd, d in means.items():
        print(f"  {rd}: {len(d)} ta'amim with enough instances")

    readers = sorted(means)
    print(f"\n{'pair':<12}{'shared ta_amim':>16}{'contour distance':>19}")
    print("-" * 48)
    floors = {}
    for rd in readers:
        f, n = within_reader_floor(store, rd)
        if f is not None:
            floors[rd] = f
            print(f"{rd + ' vs self':<12}{n:>16}{f:>19.3f}")
    print("-" * 48)
    pairs = {}
    for i in range(len(readers)):
        for j in range(i + 1, len(readers)):
            a, b = readers[i], readers[j]
            d, n = reader_distance(means[a], means[b])
            if d is None:
                continue
            pairs[(a, b)] = d
            print(f"{a + ' vs ' + b:<12}{n:>16}{d:>19.3f}")
    print("-" * 48)

    floor = np.mean(list(floors.values())) if floors else float("nan")
    print(f"within-reader floor (sampling noise): {floor:.3f}")
    print("A pair at the floor is indistinguishable; the gap above it is the real\n"
          "melodic difference between those two readers.")

    same = [d for (a, b), d in pairs.items() if "R6" not in (a, b)]
    cross = [d for (a, b), d in pairs.items() if "R6" in (a, b)]
    if same and cross:
        s, c = np.mean(same), np.mean(cross)
        print(f"\nR1-R2 (same tradition by label) : {s:.3f}  "
              f"= floor + {s - floor:+.3f}")
        print(f"pairs involving R6              : {c:.3f}  "
              f"= floor + {c - floor:+.3f}")
        if c > s:
            print(f"\nR6 is {(c - floor) / max(s - floor, 1e-9):.1f}x further from the "
                  f"Sephardic readers, above the noise floor,\nthan they are from each "
                  f"other. The grouping is acoustic, not just a label.")
        else:
            print("\nThe labelled grouping is NOT reproduced acoustically. Every claim\n"
                  "resting on 'R1 and R2 share a tradition' has to be withdrawn.")

    out = PROJECT_DIR / "results" / "v3_comparison" / "tradition_distance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"floors": floors,
                   "pairs": {f"{a}|{b}": d for (a, b), d in pairs.items()}}, f, indent=1)
    print(f"\nwrote {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
