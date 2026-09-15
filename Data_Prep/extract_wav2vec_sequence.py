"""
wav2vec2 as a real sequence, not a mean-pooled vector.
========================================================
The fair mean-pool retry (extract_wav2vec_meanpool.py) removed the PCA
crush but still lost to the mel-spectrogram CNN on every LORO fold (51.2%
vs 43.5% mean) -- see FINDINGS.md §9. One live, stated caveat on that
result: mean-pooling over the whole word throws away all temporal shape,
which the melody-curve/DTW work (Milestones 8-9) already showed carries
real class-discriminating information. This extracts the actual frame
sequence instead of collapsing it, so a small recurrent head can learn
from the shape, not just the average.

Fixed-length by interpolation (same convention as extract_wav2vec_features.py's
to_frames()) rather than padding, so a short and a long word's melody both
map onto the same number of steps -- comparable to the 60-point melody
curves used elsewhere in this project. float16 storage: at N_FRAMES=30,
full 80k-word corpus is ~5GB, small enough to just load into RAM whole for
training instead of building a DataLoader/disk-streaming path.

Usage:
  python Data_Prep/extract_wav2vec_sequence.py --readers R1 R2 R8
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
OUT_DIR = PROJECT_DIR / "features_w2v_seq"
MODEL_DIR = PROJECT_DIR / "models" / "wav2vec2-he"
SR = 16000
N_FRAMES = 30
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "3"))

_model = None


def _get_model():
    global _model
    if _model is None:
        from transformers import Wav2Vec2Model
        # One thread per worker. torch defaults to num_threads == core count,
        # so N workers each spawning 10 threads on a 10-core machine
        # oversubscribes ~Nx and the pool runs SLOWER than fewer workers would.
        # With 1 thread each, worker count maps directly onto real cores.
        torch.set_num_threads(1)
        _model = Wav2Vec2Model.from_pretrained(str(MODEL_DIR)).eval().to("cpu")
    return _model


def to_fixed_frames(h: np.ndarray) -> np.ndarray:
    """(T, 1024) -> (N_FRAMES, 1024) by linear interpolation along time."""
    t = np.arange(h.shape[0])
    target = np.linspace(0, max(h.shape[0] - 1, 0), N_FRAMES)
    return np.stack([np.interp(target, t, h[:, d]) for d in range(h.shape[1])], axis=1)


def process_one(r):
    import librosa
    model = _get_model()
    p = PROJECT_DIR / r["audio_file"]
    if not p.exists():
        return None
    y, _ = librosa.load(str(p), sr=SR, offset=max(0, r["start"]),
                        duration=max(0.05, r["end"] - r["start"]))
    if len(y) < SR * 0.05:
        return None
    with torch.inference_mode():
        x = torch.from_numpy(y).float().unsqueeze(0)
        h = model(x).last_hidden_state[0].numpy()   # (T, 1024)
    if len(h) < 2:
        return None
    seq = to_fixed_frames(h).astype(np.float16)      # (N_FRAMES, 1024)
    return seq, {"taam": r["taam"], "word_clean": r["word_clean"],
                "reader": r["reader"], "verse_ref": r["verse_ref"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["R1", "R2", "R8"])
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] in args.readers]
    print(f"{len(records):,} words across {args.readers}, {WORKERS} workers, "
          f"N_FRAMES={N_FRAMES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for reader in args.readers:
        rs = [r for r in records if r["reader"] == reader]
        X, meta = [], []
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(process_one, r) for r in rs]
            for n, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                if res:
                    X.append(res[0])
                    meta.append(res[1])
                if n % 2000 == 0:
                    print(f"  {reader} {n}/{len(rs)}  kept={len(X)}", flush=True)
        if not X:
            print(f"{reader}: nothing extracted")
            continue
        arr = np.stack(X)   # (N, N_FRAMES, 1024) float16
        np.save(OUT_DIR / f"{reader}_X.npy", arr)
        with open(OUT_DIR / f"{reader}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"{reader}: {arr.shape} ({arr.nbytes/1e9:.2f}GB) -> "
              f"{OUT_DIR.name}/{reader}_X.npy", flush=True)


if __name__ == "__main__":
    main()
