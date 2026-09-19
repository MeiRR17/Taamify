"""
Per-speaker CMVN: strip the voice, keep the melody.
====================================================
Measured at matched sample rate, a model trained on Ben13 scores 5.6% on R1
and 5.5% on R2 -- chance for 21 classes is 4.8%. It is not a labelling bug:
on its own voice the model's prediction distribution tracks the truth almost
exactly, while on another voice it collapses onto a single class (Qadma, 54%
and 72%, against a true prior of under 3%). That is what a model looks like
when it has learned a speaker rather than a cantillation system.

The first and cheapest thing to try is to take the speaker out of the
features. A log-mel spectrogram carries two things at once: the melodic
contour, which is what a ta'am is, and the speaker's vocal tract and pitch
range, which is what the model appears to have latched onto. Subtracting each
speaker's own mean and dividing by their own standard deviation, per mel band,
removes the part that is constant about a voice and leaves the part that
varies within it.

This is standard CMVN, and it is not a guaranteed fix -- it removes the
speaker's average spectrum, not their pitch range or their timing. It is
simply the measurement that costs an hour instead of a month, so it goes
first.

Padding is preserved: frames that are entirely silent stay at the pad value
rather than being normalized into noise.

Usage:
  python Data_Prep/make_speaker_normalized.py --reader BEN13R
  python Data_Prep/make_speaker_normalized.py --reader BEN13R R1 R2
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
SRC = P / "features_v3_word"
DST = P / "features_v3_word_spknorm"
PAD = -80.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", nargs="+", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)

    for r in args.reader:
        x_path = SRC / f"{r}_X.npy"
        if not x_path.exists():
            print(f"  {r}: אין קובץ, מדלג")
            continue
        X = np.load(x_path, mmap_mode="r")
        n, mels, frames = X.shape

        # two passes so the whole array never has to be resident
        total = np.zeros(mels, dtype=np.float64)
        total_sq = np.zeros(mels, dtype=np.float64)
        count = np.zeros(mels, dtype=np.float64)
        for a in range(0, n, args.chunk):
            s = np.asarray(X[a:a + args.chunk], dtype=np.float32)
            live = ~(s <= PAD + 0.1).all(axis=1)          # (chunk, frames)
            m = np.broadcast_to(live[:, None, :], s.shape)
            total += (s * m).sum(axis=(0, 2))
            total_sq += ((s ** 2) * m).sum(axis=(0, 2))
            count += m.sum(axis=(0, 2))
        mean = total / np.maximum(count, 1)
        var = total_sq / np.maximum(count, 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-6))

        out = np.lib.format.open_memmap(DST / f"{r}_X.npy", mode="w+",
                                        dtype=np.float32,
                                        shape=(n, mels, frames))
        for a in range(0, n, args.chunk):
            s = np.asarray(X[a:a + args.chunk], dtype=np.float32)
            live = ~(s <= PAD + 0.1).all(axis=1)
            z = (s - mean[None, :, None]) / std[None, :, None]
            # keep the padding recognisable rather than normalising silence
            z = np.where(live[:, None, :], z, PAD)
            out[a:a + args.chunk] = z
        out.flush()
        shutil.copyfile(SRC / f"{r}_meta.json", DST / f"{r}_meta.json")
        print(f"  {r}: {n:,} שורות · ממוצע הקורא הוסר "
              f"(טווח {mean.min():.1f}..{mean.max():.1f} dB)")


if __name__ == "__main__":
    main()
