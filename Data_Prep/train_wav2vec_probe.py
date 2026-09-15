"""
Fair wav2vec2 linear probe, trained standalone (no shared pipeline files).
===========================================================================
The first wav2vec2 attempt (extract_wav2vec_features.py + 10_train_pipeline.py
--model mlp) forced the model's 1024-dim hidden state through a PCA crush to
128 dims so it would fit DeepCNN/MLP's hardcoded `Linear(128*87, ...)` input
shape (both classes assume a 128x87 mel-spectrogram). Result: 22-27% accuracy
-- a collapse. That is evidence the 128-dim bottleneck destroys the
representation, not evidence about wav2vec2 itself.

This is the standard "linear probe" methodology from the self-supervised
speech literature: a small classifier trained directly on the model's own
mean-pooled 1024-dim embedding (from extract_wav2vec_meanpool.py), no lossy
projection. Deliberately NOT built on 00_validate_and_prepare_data.py /
10_train_pipeline.py -- both are hardcoded to the 128x87 spectrogram shape
and are shared, actively-running files (the phrase-realignment sweep uses
them concurrently); duplicating the ~40 lines of split/train logic here
avoids touching either while that sweep is in flight.

Same evaluation discipline as everywhere else in the project: LORO by
reader (train on two, test on the third's held-out original alignment),
top-11 classes, same list `00_validate_and_prepare_data.py --num-classes 11`
selects. Reader-level split has no leakage risk (unlike a verse-level
split), so no verse bookkeeping is needed here.

Usage:
  python Data_Prep/train_wav2vec_probe.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEAT_DIR = PROJECT_DIR / "features_w2v_pool"
OUT_DIR = PROJECT_DIR / "results" / "wav2vec_probe"
READERS = ["R1", "R2", "R8"]
TOP11 = ['Tipecha', 'Munach', 'Mercha', 'Zaqef_Qatan', 'Etnachta', 'Pashta',
         'Siluk', 'Mahapakh', 'Tevir', 'Revia', 'Qadma']
SEED = 42
EPOCHS = 60
PATIENCE = 10


class Probe(nn.Module):
    """Small MLP head, same family (BatchNorm+Dropout) as 10_train_pipeline.py's
    own MLP baseline, just with a 1024-dim input instead of 128*87."""
    def __init__(self, in_dim=1024, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_reader(reader):
    X = np.load(FEAT_DIR / f"{reader}_X.npy")
    with open(FEAT_DIR / f"{reader}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    keep = [i for i, m in enumerate(meta) if m["taam"] in TOP11]
    return X[keep], [meta[i]["taam"] for i in keep]


def train_fold(holdout, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_readers = [r for r in READERS if r != holdout]
    Xtr = np.concatenate([load_reader(r)[0] for r in train_readers])
    ytr_taam = sum((load_reader(r)[1] for r in train_readers), [])
    Xte, yte_taam = load_reader(holdout)

    cls_to_idx = {c: i for i, c in enumerate(TOP11)}
    ytr = np.array([cls_to_idx[t] for t in ytr_taam])
    yte = np.array([cls_to_idx[t] for t in yte_taam])

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    n_val = max(200, int(0.1 * len(Xtr)))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xtr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xva, yva = Xtr[val_idx], ytr[val_idx]
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]

    counts = np.bincount(ytr, minlength=len(TOP11))
    weights = torch.tensor(counts.sum() / (counts + 1), dtype=torch.float32)
    weights = weights / weights.mean()

    model = Probe(Xtr.shape[1], len(TOP11)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit = nn.CrossEntropyLoss(weight=weights.to(device))

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)
    yva_t = torch.from_numpy(yva).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)

    best_val, best_state, bad = -1.0, None, 0
    bs = 128
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t).argmax(1).cpu().numpy()
        val_acc = accuracy_score(yva, val_pred)
        sched.step(val_acc)
        if val_acc > best_val:
            best_val, best_state, bad = val_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte_t).cpu().numpy()
    te_pred = te_logits.argmax(1)

    metrics = {
        "accuracy": float(accuracy_score(yte, te_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, te_pred)),
        "mcc": float(matthews_corrcoef(yte, te_pred)),
        "f1_macro": float(f1_score(yte, te_pred, average="macro")),
        "f1_weighted": float(f1_score(yte, te_pred, average="weighted")),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "best_val_accuracy": float(best_val),
    }
    return metrics


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for holdout in READERS:
        print(f"=== LORO fold: {holdout} held out (device={device}) ===", flush=True)
        m = train_fold(holdout, device)
        results[holdout] = m
        print(f"  {holdout}: acc={m['accuracy']:.4f}  balanced={m['balanced_accuracy']:.4f}  "
              f"mcc={m['mcc']:.4f}  (n_test={m['n_test']})", flush=True)
        with open(OUT_DIR / f"loro_{holdout}_metrics.json", "w") as f:
            json.dump(m, f, indent=2)

    mean_acc = np.mean([results[r]["accuracy"] for r in READERS])
    print(f"\nmean LORO accuracy: {mean_acc:.4f}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"per_fold": results, "mean_accuracy": float(mean_acc)}, f, indent=2)
    print(f"saved: {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
