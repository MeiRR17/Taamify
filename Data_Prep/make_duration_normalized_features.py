"""
Duration-normalised features: test whether R1's wall is a tempo artifact.
==========================================================================
THE FINDING THAT MOTIVATES THIS (Data_Prep/diagnose_r1_wall.py, run
2026-09-08, results/r1_wall/diagnosis.txt):

  reader   median word duration   between-class duration separability (F)
  R1       0.940 s                404.40
  R2       0.860 s                195.17
  R8       0.860 s                130.20

R1 reads SLOWER, and its ta'amim are separated by duration more than twice
as strongly as R8's. That reverses the usual reading of the R1 wall. R1 is
not the noisy reader -- on the one property measurable without any model at
all, R1's classes are the *most* distinguishable. What makes R1 hard is
therefore not that its audio is worse, but that its timing distribution is
systematically DIFFERENT: a model trained on R2+R8 learns their
duration->class mapping and then meets a reader whose mapping is stretched.

That is a distribution-shift hypothesis, and it is directly testable: strip
absolute duration out of the features entirely and see whether the R1 fold
specifically improves.

HOW: every word is already a 128x87 mel-dB image, right-padded with silence
(PAD_DB) when the word is shorter than 87 frames -- so absolute duration is
encoded as "how many of the 87 columns are real". This finds each word's
real (non-pad) region and resizes it to fill all 87 columns, so every word
occupies the full width regardless of how long it actually was. Shape stays
128x87, so the whole downstream pipeline is unchanged.

Deliberately operates on the EXISTING extracted features rather than
re-processing audio: it is pure numpy, needs no model, and takes minutes.

Usage:
  python Data_Prep/make_duration_normalized_features.py --readers R1 R2 R8
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "features_v3_word"
OUT_DIR = PROJECT_DIR / "features_v3_dnorm"
PAD_DB = -80.0
MAX_FRAMES = 87


def real_width(spec: np.ndarray) -> int:
    """Number of leading columns that aren't pure padding."""
    pad = np.all(spec <= PAD_DB + 0.1, axis=0) | np.all(spec == 0, axis=0)
    real = np.where(~pad)[0]
    return int(real[-1] + 1) if len(real) else spec.shape[1]


def stretch(spec: np.ndarray) -> np.ndarray:
    """Resize the real region to fill all MAX_FRAMES columns."""
    w = real_width(spec)
    if w <= 1:
        return spec
    if w == MAX_FRAMES:
        return spec
    src = spec[:, :w]
    x_old = np.arange(w)
    x_new = np.linspace(0, w - 1, MAX_FRAMES)
    return np.stack([np.interp(x_new, x_old, src[m]) for m in range(src.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["R1", "R2", "R8"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for reader in args.readers:
        x_path = SRC_DIR / f"{reader}_X.npy"
        if not x_path.exists():
            print(f"{reader}: no features, skipping")
            continue
        X = np.load(x_path)
        widths = []
        out = np.empty_like(X)
        for i in range(len(X)):
            widths.append(real_width(X[i]))
            out[i] = stretch(X[i])
        np.save(OUT_DIR / f"{reader}_X.npy", out.astype(np.float32))
        shutil.copy(SRC_DIR / f"{reader}_meta.json", OUT_DIR / f"{reader}_meta.json")
        w = np.array(widths)
        print(f"{reader}: {X.shape} -> duration-normalised  "
              f"(real width was median={np.median(w):.0f} p10={np.percentile(w,10):.0f} "
              f"p90={np.percentile(w,90):.0f} of {MAX_FRAMES})", flush=True)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
