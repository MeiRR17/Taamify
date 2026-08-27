"""
Layer Ben13's isolated single-ta'am clips onto the practice app's teacher clips.
=================================================================================
extract_teacher_examples.py builds the baseline golden-teacher clips by
cutting words out of a live R1 reading — real audio, but noisier and less
"textbook" than a purpose-made teaching demo. Ben13 (see download_ben13.py)
provides professionally produced, isolated single-ta'am demo clips for a
subset of classes; where one exists, it replaces the R1-derived clips for
that class (Ben13's is the cleaner reference); where it doesn't, the
R1-derived clips are left untouched.

Only clips whose *entire* audio demonstrates exactly one ta'am are used here
(taam_sequence of length 1) — a combination clip like "Mercha Tipcha Munach
Etnachta" can't be labelled as any single one of those classes without
segmenting it first, which this script does not attempt.

Run download_ben13.py and extract_teacher_examples.py before this.

Usage:
  python Data_Prep/build_ben13_teacher_clips.py
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_teacher_examples import (  # noqa: E402
    OUT_DIR, OUT_JSON, SR, f0_semitone_curve,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEN13_MANIFEST = PROJECT_DIR / "data" / "ben13" / "teamimtora" / "manifest.json"


def main():
    with open(BEN13_MANIFEST, encoding="utf-8") as f:
        entries = json.load(f)

    isolated = [e for e in entries if len(e["taam_sequence"]) == 1]
    print(f"{len(isolated)} isolated single-ta'am Ben13 clips found")

    with open(OUT_JSON, encoding="utf-8") as f:
        out = json.load(f)

    for e in isolated:
        cls = e["taam_sequence"][0]
        clip_id = f"{cls}_ben13"
        src = PROJECT_DIR / e["local_path"]
        y, _ = librosa.load(src, sr=SR, mono=True)
        curve, voiced_ratio = f0_semitone_curve(y, SR)
        if curve is None:
            print(f"  SKIP {cls}: pyin found too little voiced signal")
            continue

        # Drop this class's R1-derived clips — Ben13's is the cleaner,
        # canonical reference, so it replaces rather than supplements them.
        replaced = [k for k, v in out.items()
                    if v["taam"] == cls and k != clip_id]
        for k in replaced:
            (OUT_DIR / f"{k}.wav").unlink(missing_ok=True)
            del out[k]

        sf.write(OUT_DIR / f"{clip_id}.wav", y, SR)
        out[clip_id] = {
            "taam": cls,
            "word": e["text"],
            "verse_ref": "ben13.co.il reference clip",
            "duration": round(len(y) / SR, 2),
            "voiced_ratio": round(voiced_ratio, 2),
            "f0_semitones": np.round(curve, 3).tolist(),
            "source": "ben13",
        }
        note = f" (replaced {len(replaced)} R1 clip(s))" if replaced else " (new class)"
        print(f"  {cls:16s} <- {e['url']}{note}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"-> {OUT_JSON} ({len(out)} clips total)")


if __name__ == "__main__":
    main()
