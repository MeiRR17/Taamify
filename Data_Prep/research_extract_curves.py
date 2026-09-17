"""
Extract pitch contours for the MATCHED cross-reader word index.
===============================================================
Same tracker and same normalisation convention as
Data_Prep/extract_melody_curves.py (librosa.pyin, semitones relative to the
word's own median, so a curve is melodic SHAPE not absolute pitch), with three
deliberate differences:

 1. It reads results/melody_research/matched_index.json, so every curve has
    sibling curves of the SAME WORD from other readers. Word length, stress
    position and position-in-verse are held fixed across readers by
    construction rather than hoped to average out.
 2. It keeps the NATIVE-resolution track (10 ms hop), not only the 60-point
    resample, so downstream feature work is not locked into one time grid.
 3. It keeps the word's own median F0 in Hz, so a reader-level tonic can be
    subtracted later -- the per-word median normalisation is convenient but it
    throws away exactly the information needed to ask "does this ta'am end on
    the reader's tonic".

Usage:
  venv/bin/python Data_Prep/research_extract_curves.py --per-taam 400
"""
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX = PROJECT_DIR / "results" / "melody_research" / "matched_index.json"
OUT = PROJECT_DIR / "results" / "melody_research" / "curves_matched.json"
SR = 16000
HOP = 160                 # 10 ms
N_POINTS = 60
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "3"))
SEED = 0


def track(audio_path, start, end):
    try:
        y, sr = librosa.load(str(PROJECT_DIR / audio_path), sr=SR,
                             offset=max(0.0, start - 0.05),
                             duration=(end - start) + 0.1)
    except Exception:
        return None
    if len(y) < SR * 0.15:
        return None
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr,
                            frame_length=1024, hop_length=HOP)
    voiced = ~np.isnan(f0)
    if voiced.sum() < 8:
        return None
    t = np.arange(len(f0))
    filled = np.interp(t, t[voiced], f0[voiced])
    med_hz = float(np.median(f0[voiced]))
    semis = 12 * np.log2(filled / med_hz)
    return {
        "native": [round(float(x), 3) for x in semis],
        "curve": [round(float(x), 3) for x in
                  np.interp(np.linspace(0, len(semis) - 1, N_POINTS), t, semis)],
        "med_hz": round(med_hz, 2),
        "voiced_frac": round(float(voiced.mean()), 3),
        "dur": round(float(len(y) / SR), 3),
    }


def do_slot(slot):
    out = []
    for rd, r in slot["readers"].items():
        tr = track(r["audio_file"], r["start"], r["end"])
        if tr is None:
            continue
        out.append({**tr, "reader": rd, "taam": slot["taam"],
                    "ref": slot["ref"], "widx": slot["widx"],
                    "word_clean": slot["word_clean"],
                    "word_raw": slot["word_raw"], "stress": slot["stress"],
                    "n_words_in_verse": slot["n_words_in_verse"],
                    "low_confidence": r.get("low_confidence", False),
                    "ctc_score": r.get("ctc_score"),
                    "slot_id": f"{slot['ref']}|{slot['widx']}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-taam", type=int, default=400)
    ap.add_argument("--classes", nargs="+", default=None)
    args = ap.parse_args()

    slots = json.loads(INDEX.read_text(encoding="utf-8"))
    by_taam = {}
    for s in slots:
        by_taam.setdefault(s["taam"], []).append(s)

    rng = random.Random(SEED)
    picked = []
    for taam, ss in sorted(by_taam.items()):
        if args.classes and taam not in args.classes:
            continue
        # prefer slots covered by more readers -- they carry the most
        # cross-reader information per unit of extraction time
        rng.shuffle(ss)
        ss.sort(key=lambda s: -len(s["readers"]))
        picked.extend(ss[: args.per_taam])
    print(f"{len(picked):,} slots -> "
          f"{sum(len(s['readers']) for s in picked):,} curves, {WORKERS} workers")

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(do_slot, picked, chunksize=16):
            rows.extend(res)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(picked)}  curves={len(rows):,}", flush=True)

    OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"\n{len(rows):,} curves -> {OUT.relative_to(PROJECT_DIR)}")
    print("  by reader " + str(Counter(r["reader"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
