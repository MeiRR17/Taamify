"""
Extract real, per-class pitch contours -- so the melody can be SEEN, not
just trusted from a classifier's score.
=======================================================================
Direct precedent for this method: Ben-Shalom, Keshet & Yeger-Granot,
"Automatic Analysis of Music: Performance of Cantillation Signs in
Yemenite Jewish Traditional Cantillation" (CIM14, 2014) -- align text to
audio, extract each accent's pitch contour with a dedicated F0 tracker
(they used SWIPE), and look at whether the contour SHAPE is stable across
independent performances. That paper analyzed one sign (Sof Pasuq) across
two readers by hand-picked chapters. This script does the same kind of
extraction but at corpus scale: every class with real training volume,
many independent word instances, across all three active readers.

F0 TRACKER CHOICE
------------------
Tried torchcrepe (a neural F0 tracker -- the closer analogue to what a
company doing this professionally today would reach for over classical
SWIPE/YIN). It works and is a real option, but measured at ~3.2s/word
even on its smallest model on CPU -- too slow to extract hundreds of
words in one pass here. Used librosa.pyin instead: a probabilistic
extension of YIN, the same tracker already used for the teacher-clip
curves and Stage A's DTW scoring elsewhere in this project (see
webapp/server.py's f0_curve/dtw_component), so this stays consistent with
what the live app already does. Both are legitimate professional choices;
CREPE is documented here as the tested, available upgrade if per-word
accuracy ever matters more than throughput.

REPRESENTATION
---------------
Each contour is time-normalized to a fixed 60 points (interpolated over
the word's own duration, not padded) and expressed as SEMITONES relative
to that word's own median pitch -- so a curve encodes melodic SHAPE, not
absolute pitch, which varies enormously by reader and even by the same
reader's register on a given day. This is the same convention already
used by get_teacher()'s curves in webapp/server.py; extended here to many
real corpus instances instead of one demo clip per class.

Usage:
  python Data_Prep/extract_melody_curves.py                 # all classes
  python Data_Prep/extract_melody_curves.py --classes Tipecha Munach
"""
import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
OUT = PROJECT_DIR / "app_assets" / "melody_curves.json"

READERS = ["R1", "R2", "R8"]     # active Sephardic-Jerusalem tradition only
SR = 16000
N_POINTS = 60
MAX_PER_CLASS = 18
SEED = 0


def word_curve(audio_path: str, start: float, end: float):
    y, sr = librosa.load(audio_path, sr=SR, offset=max(0, start - 0.05),
                         duration=(end - start) + 0.1)
    if len(y) < SR * 0.15:
        return None
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr,
                            frame_length=1024, hop_length=160)
    voiced = ~np.isnan(f0)
    if voiced.sum() < 6:
        return None
    t = np.arange(len(f0))
    filled = np.interp(t, t[voiced], f0[voiced])
    semis = 12 * np.log2(filled / 55.0)
    semis -= np.median(semis[voiced])
    return np.interp(np.linspace(0, len(semis) - 1, N_POINTS), t, semis).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=None)
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] in READERS]

    by_class = {}
    for r in records:
        by_class.setdefault(r["taam"], []).append(r)

    classes = args.classes or sorted(
        c for c, rs in by_class.items() if len(rs) >= 5)

    rng = random.Random(SEED)
    out = {}
    for cls in classes:
        pool = by_class.get(cls, [])
        rng.shuffle(pool)
        curves = []
        for r in pool:
            if len(curves) >= MAX_PER_CLASS:
                break
            audio_path = PROJECT_DIR / r["audio_file"]
            if not audio_path.exists():
                continue
            c = word_curve(str(audio_path), r["start"], r["end"])
            if c is None:
                continue
            curves.append({"curve": [round(x, 3) for x in c],
                           "reader": r["reader"], "word": r["word_clean"],
                           "verse_ref": r["verse_ref"],
                           "duration": round(r["end"] - r["start"], 3),
                           # provenance, so the exact audio slice can be
                           # served back as a reference example
                           "audio_file": r["audio_file"],
                           "start": round(r["start"], 3),
                           "end": round(r["end"], 3)})
        out[cls] = curves
        readers_used = sorted({c["reader"] for c in curves})
        print(f"  {cls:<18}{len(curves):>3} curves  (pool {len(pool)}, "
              f"readers {readers_used})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n_total = sum(len(v) for v in out.values())
    print(f"\n{n_total} curves across {len(out)} classes -> "
          f"{OUT.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
