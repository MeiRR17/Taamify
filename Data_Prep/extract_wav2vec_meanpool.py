"""
wav2vec2 mean-pooled embeddings — the fair version.
====================================================
The first wav2vec2 attempt (extract_wav2vec_features.py) crushed the
model's real 1024-dim hidden state down to 128 via PCA so it would fit
DeepCNN's hardcoded `Linear(256*8*5, ...)` input shape. It scored 22-27% —
a collapse, but not evidence against wav2vec2 itself: forcing a
self-supervised model's representation through a 128-dim bottleneck built
for mel-spectrograms is not a fair test of it.

This is the standard "linear probe" methodology from the self-supervised
speech literature (e.g. the SUPERB benchmark): mean-pool the model's own
hidden states over time to get one 1024-dim vector per word, at full
resolution, no lossy projection. Paired with train_wav2vec_probe.py, which
trains a small classifier directly on that 1024-dim space instead of
forcing it through an architecture built for a different input shape.

Usage:
  python Data_Prep/extract_wav2vec_meanpool.py --readers R1 R2 R8
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
OUT_DIR = PROJECT_DIR / "features_w2v_pool"
MODEL_DIR = PROJECT_DIR / "models" / "wav2vec2-he"
SR = 16000
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "3"))

_model = None
_device = None


def _get_model():
    global _model, _device
    if _model is None:
        from transformers import Wav2Vec2Model
        _device = "cpu"   # MPS inside a worker process is unreliable; CPU
                          # inference on a single word (<2s audio) is fast
                          # enough that this doesn't matter for throughput
        _model = Wav2Vec2Model.from_pretrained(str(MODEL_DIR)).eval().to(_device)
    return _model, _device


def process_one(r):
    import librosa
    model, device = _get_model()
    p = PROJECT_DIR / r["audio_file"]
    if not p.exists():
        return None
    y, _ = librosa.load(str(p), sr=SR, offset=max(0, r["start"]),
                        duration=max(0.05, r["end"] - r["start"]))
    if len(y) < SR * 0.05:
        return None
    with torch.inference_mode():
        x = torch.from_numpy(y).float().unsqueeze(0).to(device)
        h = model(x).last_hidden_state[0]        # (T, 1024)
    vec = h.mean(dim=0).cpu().numpy().astype(np.float32)   # (1024,)
    return vec, {"taam": r["taam"], "word_clean": r["word_clean"],
                "reader": r["reader"], "verse_ref": r["verse_ref"],
                "start": r["start"], "end": r["end"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["R1", "R2", "R8"])
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] in args.readers]
    print(f"{len(records):,} words across {args.readers}, {WORKERS} workers")

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
        arr = np.stack(X)
        np.save(OUT_DIR / f"{reader}_X.npy", arr)
        with open(OUT_DIR / f"{reader}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"{reader}: {arr.shape} -> {OUT_DIR.name}/{reader}_X.npy")


if __name__ == "__main__":
    main()
