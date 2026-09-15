"""
Sequence-aware wav2vec2 probe: BiLSTM + attention pooling over real frames.
=============================================================================
Follow-up to the mean-pool probe (train_wav2vec_probe.py, 43.5% mean LORO,
below the 51.2% mel-spectrogram CNN baseline -- FINDINGS.md §9). The stated,
untested caveat on that result was that mean-pooling throws away all
temporal shape. This consumes extract_wav2vec_sequence.py's real
(30, 1024) frame sequences per word instead of one pooled vector, through a
small BiLSTM + learned attention pooling (not just mean-over-time, which
would reduce to a slightly smarter version of the same collapse).

Same discipline as every other probe in this project: LORO by reader,
top-11 classes (list read from prepared_data/random_top11/metadata.json,
guaranteed identical to every other number in the project), standalone (no
shared pipeline files touched).

Usage:
  python Data_Prep/train_wav2vec_seq_probe.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEAT_DIR = PROJECT_DIR / "features_w2v_seq"
OUT_DIR = PROJECT_DIR / "results" / "wav2vec_seq_probe"
READERS = ["R1", "R2", "R8"]
TOP11 = ['Tipecha', 'Munach', 'Mercha', 'Zaqef_Qatan', 'Etnachta', 'Pashta',
         'Siluk', 'Mahapakh', 'Tevir', 'Revia', 'Qadma']
SEED = 42
EPOCHS = 50
PATIENCE = 8
BATCH = 96


class SeqProbe(nn.Module):
    """Per-frame projection -> BiLSTM -> learned attention pooling -> head.
    Attention pooling (not mean-over-time) so the model can weight the
    syllable(s) that actually carry the ta'am's melody, the same intuition
    behind Milestone 9's stressed-syllable-window finding."""
    def __init__(self, in_dim=1024, num_classes=11, proj=192, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.ReLU())
        self.lstm = nn.LSTM(proj, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                      # x: (B, T, 1024)
        h = self.proj(x)                        # (B, T, proj)
        out, _ = self.lstm(h)                    # (B, T, 2*hidden)
        w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        pooled = (out * w).sum(dim=1)            # (B, 2*hidden)
        return self.head(pooled)


def load_reader(reader):
    X = np.load(FEAT_DIR / f"{reader}_X.npy")   # (N, 30, 1024) float16
    with open(FEAT_DIR / f"{reader}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    keep = [i for i, m in enumerate(meta) if m["taam"] in TOP11]
    return X[keep], [meta[i]["taam"] for i in keep]


def run_epoch(model, X, y, opt, crit, device, train, bs=BATCH):
    model.train(train)
    idx = torch.randperm(len(X)) if train else torch.arange(len(X))
    total_loss, preds = 0.0, []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        xb = torch.from_numpy(X[b.numpy()].astype(np.float32)).to(device)
        yb = y[b].to(device)
        if train:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b)
        else:
            with torch.no_grad():
                logits = model(xb)
            preds.append(logits.argmax(1).cpu())
    if not train:
        return torch.cat(preds).numpy()
    return total_loss / len(idx)


def train_fold(holdout, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_readers = [r for r in READERS if r != holdout]
    Xtr = np.concatenate([load_reader(r)[0] for r in train_readers])
    ytr_taam = sum((load_reader(r)[1] for r in train_readers), [])
    Xte, yte_taam = load_reader(holdout)

    cls_to_idx = {c: i for i, c in enumerate(TOP11)}
    ytr_all = np.array([cls_to_idx[t] for t in ytr_taam])
    yte = np.array([cls_to_idx[t] for t in yte_taam])

    mu = Xtr.reshape(-1, Xtr.shape[-1]).astype(np.float32).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).astype(np.float32).std(0) + 1e-6
    Xtr = ((Xtr.astype(np.float32) - mu) / sd).astype(np.float16)
    Xte = ((Xte.astype(np.float32) - mu) / sd).astype(np.float16)

    n_val = max(200, int(0.1 * len(Xtr)))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xtr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xva, yva = Xtr[val_idx], ytr_all[val_idx]
    Xtr, ytr = Xtr[tr_idx], ytr_all[tr_idx]

    counts = np.bincount(ytr, minlength=len(TOP11))
    weights = torch.tensor(counts.sum() / (counts + 1), dtype=torch.float32)
    weights = (weights / weights.mean()).to(device)

    model = SeqProbe(Xtr.shape[-1], len(TOP11)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit = nn.CrossEntropyLoss(weight=weights)

    ytr_t, yva_t = torch.from_numpy(ytr).long(), torch.from_numpy(yva).long()

    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS):
        loss = run_epoch(model, Xtr, ytr_t, opt, crit, device, train=True)
        val_pred = run_epoch(model, Xva, yva_t, opt, crit, device, train=False)
        val_acc = accuracy_score(yva, val_pred)
        sched.step(val_acc)
        if val_acc > best_val:
            best_val, best_state, bad = val_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        print(f"    epoch {epoch}: loss={loss:.4f} val_acc={val_acc:.4f} best={best_val:.4f}", flush=True)

    model.load_state_dict(best_state)
    te_pred = run_epoch(model, Xte, torch.from_numpy(yte).long(), opt, crit, device, train=False)

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
