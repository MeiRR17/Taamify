"""
wav2vec2 hidden states as an alternative to mel-spectrograms.
=============================================================
WHY THIS EXPERIMENT EXISTS
--------------------------
This project's central, unsolved problem is stated in every milestone: the
model does well on a reader it has seen (77.7% random split) and badly on one
it hasn't (41.7% on R1 held out). A mel-spectrogram encodes the *speaker* --
vocal tract, register, timbre -- at least as strongly as it encodes the
melody, so a CNN trained on them has every opportunity to learn the reader
instead of the ta'am.

Self-supervised speech encoders are trained on masked prediction across many
speakers, which pushes them toward representations where phonetic and prosodic
content survives and speaker identity is comparatively suppressed. If that
holds here, the unseen-reader gap should narrow. If it doesn't, that is a real
negative result about this corpus, and worth having measured rather than
assumed.

The Hebrew wav2vec2 is already on disk (models/wav2vec2-he, 1.2GB). It is used
today only for CTC forced alignment (Data_Prep/ctc_align.py) -- never as a
feature extractor.

THE SHAPE CONSTRAINT, AND HOW IT IS HANDLED
-------------------------------------------
10_train_pipeline.py's DeepCNN hardcodes `nn.Linear(256 * 8 * 5, 512)`, which
only works for a 128x87 input. Anything else crashes. So the output here must
be exactly 128x87 to stay drop-in compatible with the whole existing pipeline
(prep script, training, LORO splits, verify heads).

wav2vec2 emits (T, 1024) per word (this is the large variant).
Two reductions get it to 128x87:

  1024 -> 128 features: PCA, fitted ONCE on a random sample of frames pooled
                        across all readers, then applied to everything. Fitting
                        per-reader would leak reader identity into the basis
                        itself and quietly defeat the purpose of the experiment.
  T   -> 87 frames    : linear interpolation along time, the same convention
                        the melody-curve extractor uses.

PCA is a real information loss and is stated rather than hidden: 128 of
1024 dimensions. The retained variance is printed so the cost is on the record.

Usage:
  python Data_Prep/extract_wav2vec_features.py --readers R1 R2 R8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
OUT_DIR = PROJECT_DIR / "features_w2v"
MODEL_DIR = PROJECT_DIR / "models" / "wav2vec2-he"

SR = 16000
N_FEAT = 128          # must match DeepCNN's expected height
N_FRAMES = 87         # must match DeepCNN's expected width
PCA_SAMPLE_WORDS = 1500
SEED = 0


def load_model():
    from transformers import Wav2Vec2Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Wav2Vec2Model.from_pretrained(str(MODEL_DIR)).eval().to(device)
    return model, device


def hidden_states(model, device, y: np.ndarray):
    """(T, 1024) hidden states for one word's audio."""
    if len(y) < SR * 0.05:
        return None
    with torch.inference_mode():
        x = torch.from_numpy(y).float().unsqueeze(0).to(device)
        out = model(x).last_hidden_state[0]
    return out.cpu().numpy()


def to_frames(h: np.ndarray) -> np.ndarray:
    """(T, N_FEAT) -> (N_FEAT, N_FRAMES) by interpolating along time."""
    t = np.arange(h.shape[0])
    target = np.linspace(0, max(h.shape[0] - 1, 0), N_FRAMES)
    return np.stack([np.interp(target, t, h[:, d]) for d in range(h.shape[1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["R1", "R2", "R8"])
    args = ap.parse_args()

    import librosa
    from sklearn.decomposition import PCA

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] in args.readers]
    by_reader = {}
    for r in records:
        by_reader.setdefault(r["reader"], []).append(r)

    model, device = load_model()
    print(f"wav2vec2 on {device}; {len(records):,} words across {args.readers}")

    # ---- fit PCA once, on frames pooled across ALL readers ----
    rng = np.random.default_rng(SEED)
    sample = list(records)
    rng.shuffle(sample)
    pool = []
    for r in sample[:PCA_SAMPLE_WORDS]:
        p = PROJECT_DIR / r["audio_file"]
        if not p.exists():
            continue
        y, _ = librosa.load(str(p), sr=SR, offset=max(0, r["start"]),
                            duration=max(0.1, r["end"] - r["start"]))
        h = hidden_states(model, device, y)
        if h is not None and len(h):
            pool.append(h[rng.integers(0, len(h), min(4, len(h)))])
    frames = np.concatenate(pool)
    print(f"PCA fit on {frames.shape[0]:,} frames x {frames.shape[1]}")
    pca = PCA(n_components=N_FEAT, random_state=SEED).fit(frames)
    print(f"retained variance: {pca.explained_variance_ratio_.sum():.3f} "
          f"({N_FEAT} of {frames.shape[1]} dims)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "pca_components.npy", pca.components_)

    for reader in args.readers:
        rs = by_reader.get(reader, [])
        X, meta = [], []
        for i, r in enumerate(rs, 1):
            p = PROJECT_DIR / r["audio_file"]
            if not p.exists():
                continue
            y, _ = librosa.load(str(p), sr=SR, offset=max(0, r["start"]),
                                duration=max(0.1, r["end"] - r["start"]))
            h = hidden_states(model, device, y)
            if h is None or not len(h):
                continue
            X.append(to_frames(pca.transform(h)).astype(np.float32))
            meta.append({"taam": r["taam"], "word_clean": r["word_clean"],
                         "parasha": r["parasha"], "verse_ref": r["verse_ref"],
                         "start": r["start"], "end": r["end"],
                         "window": "word", "align": "wav2vec2"})
            if i % 2000 == 0:
                print(f"  {reader}: {i}/{len(rs)}", flush=True)
        arr = np.stack(X)
        np.save(OUT_DIR / f"{reader}_X.npy", arr)
        with open(OUT_DIR / f"{reader}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"{reader}: {arr.shape} -> {OUT_DIR.name}/", flush=True)

    print("done")


if __name__ == "__main__":
    main()
