"""
15_binary_verify_model.py — Milestone 3, Stage A: match / no-match, not "which of 11"
=======================================================================================
The real product flow (confirmed): the user picks a parasha first, so at
inference time the expected ta'am for every word is already known from
Sefaria's tagged text. The task is not blind classification, it's
verification against a known reference — a strictly easier problem, and it
should be measured as its own number, not folded into the raw 11-way
accuracy every earlier milestone reports.

Reuses 14_sequence_model.py's embed_everything() (frozen CNN -> per-word
embedding) rather than retraining a spectrogram model from scratch. On top
of each word's embedding, a small MLP is trained to answer one question:
"does this audio match candidate class C?" — input is
[embedding ; one-hot(C)], output is a single match/no-match logit.

The corpus only contains *correct* readings (no labelled mistakes), so
training pairs are constructed directly: every real word gives one positive
pair (its own true class) and one negative pair (a different, randomly
sampled wrong class) -- a balanced verification task. This still trains the
right function (does embedding e look like class c, for any e/c pair), which
is what deployment needs regardless of how training pairs were built:
deployment always queries with the text's actual expected class, correct
most of the time and occasionally wrong when a learner errs.

Usage:
  # LORO-style, to measure generalization (as above) -- writes a report only
  python 15_binary_verify_model.py --checkpoint results/loro_R1_top11_cap40000/deep_cnn/deep_cnn_best.pth \
      --readers R1 R2 R8 --holdout R1

  # Production mode, to actually deploy Stage A: no held-out reader, trains
  # on all of --readers, saves a checkpoint the webapp can load
  python 15_binary_verify_model.py --checkpoint results/random_top11/deep_cnn/deep_cnn_best.pth \
      --readers R1 R2 R8
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)

PROJECT_DIR = Path(__file__).parent
OUT_DIR = PROJECT_DIR / "results" / "binary_verify"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VerifyHead(nn.Module):
    """[embedding ; one-hot(candidate class)] -> match logit."""
    def __init__(self, emb_dim, num_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + num_classes, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb, candidate_onehot):
        return self.net(torch.cat([emb, candidate_onehot], dim=1)).squeeze(-1)


def make_pairs(emb, y, num_classes, seed):
    """One positive (true class) + one negative (random wrong class) per word."""
    rng = np.random.default_rng(seed)
    n = len(y)
    wrong = rng.integers(0, num_classes - 1, size=n)
    wrong = wrong + (wrong >= y)  # shift past the true class, never equals y

    emb2 = np.concatenate([emb, emb])
    cand = np.concatenate([y, wrong])
    match = np.concatenate([np.ones(n), np.zeros(n)])
    perm = rng.permutation(len(emb2))
    return emb2[perm], cand[perm].astype(np.int64), match[perm].astype(np.float32)


def onehot(idx, num_classes):
    out = torch.zeros(len(idx), num_classes)
    out[torch.arange(len(idx)), idx] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--readers', nargs='+', required=True)
    ap.add_argument('--holdout', default=None,
                    help="LORO-style: train/val on the other readers, test only "
                         "on this one. --checkpoint should be the CNN trained "
                         "without this reader, matching 14_sequence_model.py's "
                         "convention, for an apples-to-apples comparison. Omit "
                         "for production mode: trains on all --readers combined "
                         "(verse-level 85/15 train/val, no held-out test reader) "
                         "and saves a checkpoint instead of a measurement report.")
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    seq_mod = load_module("sequence_model", "14_sequence_model.py")
    train_mod = seq_mod.load_train_module()
    device = train_mod.pick_device()

    production = args.holdout is None
    train_readers = list(args.readers) if production else \
        [r for r in args.readers if r != args.holdout]
    print(f"embedding train/val readers {train_readers}...")
    emb_tr, _, y_tr, verses_tr, _, class_names = seq_mod.embed_everything(
        args.checkpoint, train_readers, device, train_mod)
    num_classes = len(class_names)

    if not production:
        print(f"embedding held-out reader {args.holdout}...")
        emb_te, cnn_preds_te, y_te, _, _, _ = seq_mod.embed_everything(
            args.checkpoint, [args.holdout], device, train_mod)
        baseline_top1 = accuracy_score(y_te, cnn_preds_te)

    # verse-level train/val split, same discipline as 14_sequence_model.py:
    # no verse straddles the split
    uniq_verses = np.array(sorted(set(verses_tr)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq_verses)
    n_val = max(1, int(0.15 * len(uniq_verses)))
    val_verses = set(uniq_verses[:n_val])
    in_val = np.array([v in val_verses for v in verses_tr])

    emb_train, y_train = emb_tr[~in_val], y_tr[~in_val]
    emb_val, y_val = emb_tr[in_val], y_tr[in_val]
    print(f"words: train {len(y_train)}, val {len(y_val)}"
          + ("" if production else f", test(holdout) {len(y_te)}"))

    Xp_train, Cp_train, Mp_train = make_pairs(emb_train, y_train, num_classes, args.seed)
    Xp_val, Cp_val, Mp_val = make_pairs(emb_val, y_val, num_classes, args.seed + 1)
    if not production:
        Xp_test, Cp_test, Mp_test = make_pairs(emb_te, y_te, num_classes, args.seed + 2)

    emb_dim = emb_tr.shape[1]
    model = VerifyHead(emb_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    def to_batches(X, C, M, bs, shuffle):
        idx = np.arange(len(X))
        if shuffle:
            np.random.default_rng(args.seed).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            yield (torch.from_numpy(X[b]).float().to(device),
                   onehot(C[b], num_classes).to(device),
                   torch.from_numpy(M[b]).to(device))

    def evaluate(X, C, M):
        model.eval()
        logits_all = []
        with torch.no_grad():
            for xb, cb, _ in to_batches(X, C, M, 1024, shuffle=False):
                logits_all.append(model(xb, cb).cpu())
        probs = torch.sigmoid(torch.cat(logits_all)).numpy()
        preds = (probs >= 0.5).astype(np.float32)
        return {
            "accuracy": accuracy_score(M, preds),
            "balanced_accuracy": balanced_accuracy_score(M, preds),
            "precision": precision_score(M, preds, zero_division=0),
            "recall": recall_score(M, preds, zero_division=0),
            "f1": f1_score(M, preds, zero_division=0),
            "auc": roc_auc_score(M, probs) if len(set(M)) > 1 else float("nan"),
        }

    best_val_acc, best_state, patience, bad = 0.0, None, 5, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, cb, mb in to_batches(Xp_train, Cp_train, Mp_train, args.batch_size, shuffle=True):
            opt.zero_grad()
            loss = crit(model(xb, cb), mb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        val_metrics = evaluate(Xp_val, Cp_val, Mp_val)
        print(f"  epoch {epoch}/{args.epochs} | loss {total_loss/len(Xp_train):.4f} | "
              f"val acc {val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("  early stopping")
                break

    model.load_state_dict(best_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if production:
        ckpt_path = OUT_DIR / "production_verify_head.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": list(class_names),
            "emb_dim": emb_dim,
            "hidden": 64,
            "cnn_checkpoint": str(args.checkpoint),
            "readers": list(args.readers),
            "best_val_accuracy": best_val_acc,
        }, ckpt_path)
        print(f"\nwrote -> {ckpt_path} (best val acc {best_val_acc:.4f})")
        return

    test_metrics = evaluate(Xp_test, Cp_test, Mp_test)
    out_path = OUT_DIR / f"holdout_{args.holdout}.md"
    lines = [
        f"# Stage A (binary verify) — holdout {args.holdout}",
        "",
        f"Checkpoint: {args.checkpoint} | readers: {args.readers}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *(f"| {k} | {v:.4f} |" for k, v in test_metrics.items()),
        "",
        f"Raw 11-way top-1 accuracy on the same held-out words: "
        f"**{baseline_top1:.4f}**",
        f"Stage A binary accuracy: **{test_metrics['accuracy']:.4f}** "
        f"({'+' if test_metrics['accuracy'] >= baseline_top1 else ''}"
        f"{(test_metrics['accuracy'] - baseline_top1) * 100:.1f} points)",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote -> {out_path}")
    print(f"raw 11-way top-1: {baseline_top1:.4f} | Stage A binary: "
          f"{test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
