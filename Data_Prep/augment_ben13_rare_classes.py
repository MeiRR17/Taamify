"""
Synthesize training examples for the classes with ~no real data.
==================================================================
Real corpus counts (aligned_dataset_merged.json, all 5 readers combined):
Shalshelet=2, Yerach_Ben_Yomo=2, Mercha_Kefula=12, Qarney_Para=0 — this last
one has literally never been observed anywhere else in the project. Ben13
is the only source of even a single clean example for any of these.

This is intentionally NOT applied to any other class: one voice's ~4-second
demo is not acoustic diversity, and mixing it into training for a
well-represented class would just teach the model "Ben13's voice" as a
shortcut. It's only defensible here because the real alternative is zero,
not because it's a generally good idea (see Milestone 1B/1C in the plan).

Each class's single source clip is pitch-shifted and time-stretched into
several synthetic variants — small perturbations, since a ta'am is defined
by its *relative* melodic contour, not an absolute pitch, so this doesn't
change what's being demonstrated, just gives the CNN more than one exact
waveform to avoid trivially memorizing it.

Output matches extract_features_v3.py's exact format (128 x 87 mel-dB, same
SR/N_MELS/FMAX/PAD_DB) so it drops straight into the existing
features_v3_word/ directory as a new pseudo-reader ("BEN13AUG") that
00_validate_and_prepare_data.py can include like any other --readers entry
-- gated, not automatic: Milestone 4 (all-ta'amim expansion) decides if/when
to actually include it, this script only prepares the data.

Usage:
  python Data_Prep/augment_ben13_rare_classes.py
"""

import json
from pathlib import Path

import librosa
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = Path(__import__("os").environ.get(
    "TAAMIFY_FEATURES", PROJECT_DIR / "features_v3_word"))
SEGMENTS = PROJECT_DIR / "data" / "ben13_segments" / "manifest.json"

SR = 22050
N_MELS = 128
FMAX = 8000
MAX_FRAMES = 87
PAD_DB = -80.0

# class -> source clip. Isolated single-ta'am Ben13 downloads for the two
# that never appear inside a combo; segmented pieces (from
# segment_ben13_clips.py) for the two that only exist inside the
# Yerach_Ben_Yomo/Qarney_Para combo clip.
ISOLATED_SOURCES = {
    "Shalshelet": PROJECT_DIR / "data" / "ben13" / "teamimtora" / "22.mp3",
    "Mercha_Kefula": PROJECT_DIR / "data" / "ben13" / "teamimtora" / "21.mp3",
}

PITCH_SHIFTS = [-2, -1, 0, 1, 2]      # semitones
TIME_STRETCHES = [0.9, 1.0, 1.1]      # rate


def load_at_sr(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def to_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])),
                        mode="constant", constant_values=PAD_DB)
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    return mel_db.astype(np.float32)


def variants(y: np.ndarray):
    for shift in PITCH_SHIFTS:
        for rate in TIME_STRETCHES:
            v = y
            if shift != 0:
                v = librosa.effects.pitch_shift(v, sr=SR, n_steps=shift)
            if rate != 1.0:
                v = librosa.effects.time_stretch(v, rate=rate)
            yield v, shift, rate


def main():
    sources = {}
    for cls, path in ISOLATED_SOURCES.items():
        if path.exists():
            sources[cls] = [("isolated", path)]
        else:
            print(f"  WARNING: {cls} source {path} not found, skipping")

    if SEGMENTS.exists():
        with open(SEGMENTS, encoding="utf-8") as f:
            segs = json.load(f)
        for e in segs:
            if e["taam"] in ("Yerach_Ben_Yomo", "Qarney_Para"):
                sources.setdefault(e["taam"], []).append(
                    ("segmented", PROJECT_DIR / e["local_path"]))
    else:
        print("  WARNING: no segmented-clips manifest — run "
              "segment_ben13_clips.py first")

    X_parts, meta = [], []
    for cls, clips in sources.items():
        n = 0
        for kind, path in clips:
            y = load_at_sr(path)
            for v, shift, rate in variants(y):
                X_parts.append(to_mel(v))
                meta.append({
                    "taam": cls,
                    "source": "ben13_augmented",
                    "source_kind": kind,
                    "source_path": str(path.relative_to(PROJECT_DIR)),
                    "pitch_shift_semitones": shift,
                    "time_stretch_rate": rate,
                })
                n += 1
        print(f"  {cls:<16} {len(clips)} source clip(s) -> {n} synthetic variants")

    if not X_parts:
        print("nothing to write")
        return

    X = np.stack(X_parts)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "BEN13AUG_X.npy", X)
    with open(FEATURES_DIR / "BEN13AUG_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {X.shape} -> {FEATURES_DIR / 'BEN13AUG_X.npy'}")
    print("Not included in any training run automatically -- pass "
          "--readers ... BEN13AUG explicitly when Milestone 4 is ready to "
          "use it.")


if __name__ == "__main__":
    main()
