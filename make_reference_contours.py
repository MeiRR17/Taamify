"""
make_reference_contours.py — "golden teacher" melodic contours per ta'am
========================================================================
For the practice mode: extracts, for every ta'am class, a reference melodic
contour (median dominant-frequency curve over time) + a tolerance band,
computed from the verified dataset itself (features_v2 mel spectrograms).

Both the reference and the live student audio go through the SAME transform
(mel spectrogram -> dominant-bin contour -> semitone-normalize), so the DTW
comparison in the app is apples-to-apples.

Output: app_assets/taam_reference_contours.json
  {taam: {"median": [...], "lo": [...], "hi": [...], "n": int}}
  Curves are in semitones relative to each sample's own median pitch
  (key normalization — a child and an adult produce the same curve).

Usage: TAAMIFY_FEATURES=/tmp/features_v2 python make_reference_contours.py
"""

import json
import os
from pathlib import Path

import librosa
import numpy as np

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = Path(os.environ.get("TAAMIFY_FEATURES", PROJECT_DIR / "features_v2"))
OUT = PROJECT_DIR / "app_assets" / "taam_reference_contours.json"

SR = 22050
N_MELS = 128
FMAX = 8000
PAD_DB = -80.0
N_POINTS = 40           # contours resampled to fixed length
MAX_PER_CLASS = 800     # enough for stable medians
CLASSES = ["Tipecha", "Munach", "Mercha", "Zaqef_Qatan", "Etnachta", "Pashta",
           "Siluk", "Mahapakh", "Tevir", "Revia", "Qadma"]

MEL_FREQS = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX)
# singing F0 range: focus the argmax on bins between ~80 and 800 Hz
LO_BIN = int(np.searchsorted(MEL_FREQS, 80))
HI_BIN = int(np.searchsorted(MEL_FREQS, 800))


def spec_to_contour(spec_db: np.ndarray):
    """(128, T) dB mel -> semitone contour relative to its own median.
    Returns None if the voiced part is too short."""
    # frames that actually contain signal (not the -80 dB padding)
    frame_energy = spec_db.max(axis=0)
    voiced = frame_energy > (PAD_DB + 20)
    if voiced.sum() < 8:
        return None
    S = spec_db[LO_BIN:HI_BIN, voiced]
    # dominant melodic bin per frame, weighted parabolic-free argmax
    bins = S.argmax(axis=0) + LO_BIN
    freqs = MEL_FREQS[bins]
    freqs[freqs < 1] = 1
    semitones = 12 * np.log2(freqs)
    semitones = semitones - np.median(semitones)   # key normalization
    # clip octave-jump artifacts, then resample to fixed length
    semitones = np.clip(semitones, -12, 12)
    x_old = np.linspace(0, 1, len(semitones))
    x_new = np.linspace(0, 1, N_POINTS)
    return np.interp(x_new, x_old, semitones)


def main():
    OUT.parent.mkdir(exist_ok=True)
    refs = {}
    rng = np.random.default_rng(42)
    for reader in ["R1", "R2"]:      # the two full-Torah readers as "teachers"
        X = np.load(FEATURES_DIR / f"{reader}_X.npy", mmap_mode="r")
        with open(FEATURES_DIR / f"{reader}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        for cls in CLASSES:
            idx = [i for i, m in enumerate(meta) if m["taam"] == cls]
            if len(idx) > MAX_PER_CLASS // 2:
                idx = list(rng.choice(idx, MAX_PER_CLASS // 2, replace=False))
            curves = []
            for i in idx:
                c = spec_to_contour(np.asarray(X[i]))
                if c is not None:
                    curves.append(c)
            if curves:
                refs.setdefault(cls, []).extend(curves)
        del X
        print(f"{reader}: done", flush=True)

    out = {}
    for cls, curves in refs.items():
        A = np.stack(curves)
        out[cls] = {
            "median": np.median(A, axis=0).round(3).tolist(),
            "lo": np.percentile(A, 25, axis=0).round(3).tolist(),
            "hi": np.percentile(A, 75, axis=0).round(3).tolist(),
            "n": len(curves),
        }
        print(f"{cls:14s} {len(curves)} contours")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
