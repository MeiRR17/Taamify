"""
extract_teacher_examples.py — golden-teacher word clips + real F0 curves
========================================================================
Practice mode works like singing apps: the student sings a SPECIFIC word
after the teacher. This script cuts, for every ta'am, a few clean example
words from reader R1's bereshit recording, and extracts their true pitch
curves with pyin (not spectral argmax — that proved non-discriminative:
0.99x class separation).

Output:
  app_assets/teacher/<Taam>_<k>.wav        (small word clips, playable in app)
  app_assets/teacher_contours.json         {clip_id: {taam, word, f0_semitones,
                                            voiced_ratio, duration}}

Selection: longest-duration, highest-probability verified words of each ta'am
in 01_bereshit (single 400 MB download instead of 54).
"""

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).parent
WAV = PROJECT_DIR / "data" / "R1" / "wav" / "01_bereshit.wav"
ALIGNED = PROJECT_DIR / "aligned_dataset_v2.json"
OUT_DIR = PROJECT_DIR / "app_assets" / "teacher"
OUT_JSON = PROJECT_DIR / "app_assets" / "teacher_contours.json"

SR = 22050
PAD_SEC = 0.08
PER_CLASS = 3
N_POINTS = 60
CLASSES = ["Tipecha", "Munach", "Mercha", "Zaqef_Qatan", "Etnachta", "Pashta",
           "Siluk", "Mahapakh", "Tevir", "Revia", "Qadma"]


def f0_semitone_curve(y: np.ndarray, sr: int):
    """pyin F0 -> key-normalized semitone curve resampled to N_POINTS."""
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr,
                                 frame_length=1024, hop_length=256)
    good = ~np.isnan(f0)
    if good.sum() < 10:
        return None, 0.0
    t = np.arange(len(f0))
    f0_filled = np.interp(t, t[good], f0[good])   # bridge unvoiced gaps
    semitones = 12 * np.log2(f0_filled / 55.0)
    semitones -= np.median(semitones[good])       # key normalization
    x_new = np.linspace(0, len(semitones) - 1, N_POINTS)
    return np.interp(x_new, t, semitones), float(good.mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALIGNED, encoding="utf-8") as f:
        records = [r for r in json.load(f)
                   if r["reader"] == "R1" and r["parasha"] == "01_bereshit"]
    print(f"{len(records)} verified words in R1/bereshit")

    print(f"loading {WAV.name} (may download from Drive)...", flush=True)
    y_full, _ = librosa.load(WAV, sr=SR, mono=True)

    out = {}
    for cls in CLASSES:
        cands = [r for r in records if r["taam"] == cls]
        # prefer long, confident words — clearest melodies
        cands.sort(key=lambda r: (r["end"] - r["start"]) * r["probability"],
                   reverse=True)
        kept = 0
        for r in cands:
            if kept >= PER_CLASS:
                break
            s = max(0, int((r["start"] - PAD_SEC) * SR))
            e = min(len(y_full), int((r["end"] + PAD_SEC) * SR))
            clip = y_full[s:e]
            curve, voiced_ratio = f0_semitone_curve(clip, SR)
            if curve is None or voiced_ratio < 0.55:
                continue
            clip_id = f"{cls}_{kept}"
            sf.write(OUT_DIR / f"{clip_id}.wav", clip, SR)
            out[clip_id] = {
                "taam": cls,
                "word": r["word_clean"],
                "verse_ref": r["verse_ref"],
                "duration": round(len(clip) / SR, 2),
                "voiced_ratio": round(voiced_ratio, 2),
                "f0_semitones": np.round(curve, 3).tolist(),
            }
            kept += 1
        print(f"{cls:14s} {kept} clips")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"-> {OUT_JSON} ({len(out)} clips)")


if __name__ == "__main__":
    main()
