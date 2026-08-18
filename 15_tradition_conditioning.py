"""
15_tradition_conditioning.py — does conditioning on nusach repair naive mixing?
==============================================================================
The finding says the model learns a tradition, not the cantillation system. A
consequence showed up by accident: training on R1 alone scores 0.565 on R2,
while training on R1+R6 scores 0.472 on the same test set. Adding the Ashkenazi
reader *hurts* Sephardic performance. Two traditions assign different melodies
to the same eleven labels, so a model with one output head has to average them.

If that reading is right, telling the model which tradition it is listening to
should recover the loss. This trains four variants on identical data:

  single      train on the same-tradition reader only        (the ceiling)
  naive       train on both readers, one head                (the damage)
  tradition   both readers + a learned tradition embedding   (the proposal)
  reader      both readers + a learned reader embedding      (the ablation)

The reader ablation is the one that can falsify the framing: if conditioning on
*reader identity* works as well as conditioning on tradition, then "tradition"
is just a coarse name for speaker adaptation and the paper's framing is wrong.

Conditioning is only meaningful when the tradition is known at inference, so the
held-out reader's tradition is always represented in training by its
tradition-mate. Generalizing to an *unseen* tradition is a different problem and
is not what this measures.

Usage:
  python 15_tradition_conditioning.py --test-reader R2 --epochs 30
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_DIR = Path(__file__).resolve().parent
FEATURES = Path(os.environ.get("TAAMIFY_FEATURES", PROJECT_DIR / "features_v3_word"))

TRADITION = {"R1": "sephardic_jm", "R2": "sephardic_jm",
             "R6": "ashkenazi", "R7": "yemenite"}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionedCNN(nn.Module):
    """DeepCNN's trunk, with an optional condition embedding at the head.

    The embedding joins at the classifier rather than the input: the convolutional
    features of a sung syllable are the same physics in any tradition, what
    differs is which label they map to. Conditioning the mapping, not the
    perception.
    """

    def __init__(self, num_classes=11, n_conditions=0, cond_dim=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        self.n_conditions = n_conditions
        extra = 0
        if n_conditions:
            self.cond = nn.Embedding(n_conditions, cond_dim)
            extra = cond_dim
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 5, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5))
        self.head = nn.Sequential(
            nn.Linear(512 + extra, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes))

    def forward(self, x, c=None):
        h = self.fc1(self.features(x).flatten(1))
        if self.n_conditions:
            h = torch.cat([h, self.cond(c)], dim=1)
        return self.head(h)


def load_reader(reader):
    X = np.load(FEATURES / f"{reader}_X.npy").astype(np.float32)
    with open(FEATURES / f"{reader}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return X, np.array([m["taam"] for m in meta])


def normalize(X):
    m = X.mean(axis=(1, 2), keepdims=True)
    s = X.std(axis=(1, 2), keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def run(model, loaders, device, class_weights, epochs, lr=1e-3):
    train_loader, val_loader, test_loader = loaders
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    best_state, best_val = None, -1.0

    for ep in range(epochs):
        model.train()
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, cb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for xb, cb, yb in val_loader:
                p = model(xb.to(device), cb.to(device)).argmax(1).cpu()
                correct += (p == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1)
        if acc > best_val:
            best_val = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.inference_mode():
        for xb, cb, yb in test_loader:
            preds.append(model(xb.to(device), cb.to(device)).argmax(1).cpu())
            trues.append(yb)
    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()
    return {"accuracy": float((p == t).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(t, p)),
            "mcc": float(matthews_corrcoef(t, p)),
            "f1_macro": float(f1_score(t, p, average="macro"))}


def make_loaders(Xtr, ctr, ytr, Xva, cva, yva, Xte, cte, yte, bs):
    from torch.utils.data import TensorDataset, DataLoader
    def dl(X, c, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).unsqueeze(1),
                           torch.from_numpy(c).long(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)
    return dl(Xtr, ctr, ytr, True), dl(Xva, cva, yva, False), dl(Xte, cte, yte, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-reader", default="R2")
    ap.add_argument("--same-tradition", default="R1", help="the held-out reader's tradition-mate")
    ap.add_argument("--other-tradition", default="R6")
    ap.add_argument("--num-classes", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/v3_comparison/conditioning.json")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}   features: {FEATURES.name}")

    readers = [args.same_tradition, args.other_tradition, args.test_reader]
    data = {r: load_reader(r) for r in readers}

    counts = Counter()
    for r in readers:
        counts.update(data[r][1])
    classes = [c for c, _ in counts.most_common(args.num_classes)]
    cls_idx = {c: i for i, c in enumerate(classes)}
    print(f"classes: {classes}")

    def prep(r):
        X, y = data[r]
        keep = np.array([t in cls_idx for t in y])
        return normalize(X[keep]), np.array([cls_idx[t] for t in y[keep]])

    Xs, ys = prep(args.same_tradition)
    Xo, yo = prep(args.other_tradition)
    Xt, yt = prep(args.test_reader)
    print(f"{args.same_tradition}: {len(Xs):,}   {args.other_tradition}: {len(Xo):,}   "
          f"test {args.test_reader}: {len(Xt):,}")

    # Condition codes. For 'tradition', the held-out reader is told its own
    # tradition, which training saw through its tradition-mate. For 'reader',
    # it is told its mate's identity — the closest thing available, and exactly
    # the advantage the ablation needs in order to be a fair rival.
    trad_codes = {args.same_tradition: 0, args.other_tradition: 1}
    test_trad = 0                      # same tradition as its mate
    read_codes = {args.same_tradition: 0, args.other_tradition: 1}
    test_read = 0

    variants = {
        "single":    dict(readers=[args.same_tradition], cond=None),
        "naive":     dict(readers=[args.same_tradition, args.other_tradition], cond=None),
        "tradition": dict(readers=[args.same_tradition, args.other_tradition], cond="tradition"),
        "reader":    dict(readers=[args.same_tradition, args.other_tradition], cond="reader"),
    }

    results = {}
    for name, cfg in variants.items():
        parts_X, parts_y, parts_c = [], [], []
        for r in cfg["readers"]:
            X, y = (Xs, ys) if r == args.same_tradition else (Xo, yo)
            parts_X.append(X)
            parts_y.append(y)
            if cfg["cond"] == "tradition":
                code = trad_codes[r]
            elif cfg["cond"] == "reader":
                code = read_codes[r]
            else:
                code = 0
            parts_c.append(np.full(len(X), code))
        X = np.concatenate(parts_X)
        y = np.concatenate(parts_y)
        c = np.concatenate(parts_c)

        # Same training budget for every variant, or 'single' would simply be
        # the small-data condition and the comparison would measure size.
        if args.max_train and len(X) > args.max_train:
            idx, _ = train_test_split(np.arange(len(X)), train_size=args.max_train,
                                      random_state=args.seed, stratify=y)
            X, y, c = X[idx], y[idx], c[idx]

        Xtr, Xva, ytr, yva, ctr, cva = train_test_split(
            X, y, c, test_size=0.15, random_state=args.seed, stratify=y)

        cte = np.full(len(Xt), test_trad if cfg["cond"] == "tradition"
                      else (test_read if cfg["cond"] == "reader" else 0))
        loaders = make_loaders(Xtr, ctr, ytr, Xva, cva, yva, Xt, cte, yt, args.batch_size)

        n_cond = 0 if cfg["cond"] is None else 2
        model = ConditionedCNN(num_classes=len(classes), n_conditions=n_cond).to(device)
        w = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        cw = torch.FloatTensor(w).to(device)

        print(f"\n── {name}: train {len(Xtr):,} (cond={cfg['cond']}) ──", flush=True)
        results[name] = run(model, loaders, device, cw, args.epochs)
        print(f"   {results[name]}")

    print(f"\n{'variant':<12}{'accuracy':>10}{'balanced':>10}{'mcc':>9}{'f1_macro':>10}")
    print("-" * 51)
    for k in ("single", "naive", "tradition", "reader"):
        m = results[k]
        print(f"{k:<12}{m['accuracy']:>10.4f}{m['balanced_accuracy']:>10.4f}"
              f"{m['mcc']:>9.4f}{m['f1_macro']:>10.4f}")
    print("-" * 51)
    damage = results["single"]["accuracy"] - results["naive"]["accuracy"]
    repair = results["tradition"]["accuracy"] - results["naive"]["accuracy"]
    print(f"naive mixing costs      : {damage:+.4f}")
    print(f"tradition conditioning  : {repair:+.4f}  "
          f"({100 * repair / damage:.0f}% of the loss recovered)" if damage > 0 else "")
    print(f"reader conditioning     : "
          f"{results['reader']['accuracy'] - results['naive']['accuracy']:+.4f}")
    print("\nIf reader conditioning matches tradition conditioning, 'tradition' is a\n"
          "coarse label for speaker adaptation and the framing needs rethinking.")

    out = PROJECT_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"test_reader": args.test_reader, "seed": args.seed,
               "features": FEATURES.name, "results": results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
