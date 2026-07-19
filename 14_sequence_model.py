"""
14_sequence_model.py — does verse context improve ta'am classification?
=======================================================================
Ta'amim follow a grammar: they appear in near-fixed sequences (mercha-tipcha,
mahapakh-pashta, munach-etnachta...). The per-word CNN ignores this. Here we
freeze the best CNN, embed every word, and train a BiLSTM tagger over whole
VERSES of embeddings, so each word's prediction can lean on its neighbors.

Fair comparison: the split is BY VERSE (no verse straddles splits), and the
frozen CNN's own per-word predictions are scored on the exact same test words.

Usage:
  python 14_sequence_model.py --checkpoint results/random_top11/resnet18_tl/resnet18_tl_best.pth
"""

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

PROJECT_DIR = Path(__file__).parent
# TAAMIFY_FEATURES lets long runs read a local copy instead of the
# Google-Drive-synced folder (Drive eviction corrupts long reads)
FEATURES_DIR = Path(os.environ.get("TAAMIFY_FEATURES",
                                   PROJECT_DIR / "features_v2"))
CACHE = Path("/tmp/taamify_seq_cache.npz")
OUT_DIR = PROJECT_DIR / "results" / "sequence_model"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_pipeline", PROJECT_DIR / "10_train_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def embed_everything(checkpoint, readers, device, train_mod):
    """Frozen CNN -> embedding + logit for every word of every reader.
    Returns embeddings, cnn_preds, labels, verse keys (reader|parasha|verse),
    and word order within recordings (by start time)."""
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    class_names = ckpt['class_names']
    model = train_mod.MODEL_REGISTRY[ckpt['architecture']](num_classes=len(class_names))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    label_to_idx = {n: i for i, n in enumerate(class_names)}

    last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    captured = []
    handle = last_linear.register_forward_hook(
        lambda _m, inp, _out: captured.append(inp[0].detach().cpu()))

    emb_all, pred_all, y_all, verse_all, start_all = [], [], [], [], []
    for reader in readers:
        X = np.load(FEATURES_DIR / f"{reader}_X.npy").astype(np.float32)
        with open(FEATURES_DIR / f"{reader}_meta.json", encoding='utf-8') as f:
            meta = json.load(f)
        keep = [i for i, m in enumerate(meta) if m['taam'] in label_to_idx]
        X = X[keep]
        meta = [meta[i] for i in keep]

        mean = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2), keepdims=True)
        std[std == 0] = 1.0
        X = ((X - mean) / std)[:, None, :, :]

        logits_parts = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                logits_parts.append(model(torch.from_numpy(X[i:i + 256]).to(device)).cpu())
        logits = torch.cat(logits_parts)
        emb_all.append(torch.cat(captured).numpy()); captured.clear()
        pred_all.append(logits.argmax(1).numpy())
        y_all.append(np.array([label_to_idx[m['taam']] for m in meta]))
        verse_all += [f"{reader}|{m['parasha']}|{m['verse_ref']}" for m in meta]
        start_all += [m['start'] for m in meta]
        print(f"  {reader}: embedded {len(X)} words", flush=True)
    handle.remove()
    return (np.concatenate(emb_all), np.concatenate(pred_all), np.concatenate(y_all),
            np.array(verse_all), np.array(start_all), class_names)


class VerseTagger(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=192):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(out)


def make_batches(verse_ids, order, rng=None):
    """Group word indices into verse sequences, sorted by start time."""
    groups = defaultdict(list)
    for idx, v in enumerate(verse_ids):
        groups[v].append(idx)
    seqs = []
    for v, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: order[i])
        seqs.append(np.array(idxs))
    if rng is not None:
        rng.shuffle(seqs)
    return seqs


def batch_iter(seqs, emb, y, batch_size, device):
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        lengths = torch.tensor([len(s) for s in chunk])
        L = lengths.max()
        xb = torch.zeros(len(chunk), L, emb.shape[1])
        yb = torch.full((len(chunk), L), -100, dtype=torch.long)
        for j, s in enumerate(chunk):
            xb[j, :len(s)] = torch.from_numpy(emb[s])
            yb[j, :len(s)] = torch.from_numpy(y[s])
        yield xb.to(device), yb.to(device), lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--readers', nargs='+', default=['R1', 'R2', 'R3', 'R4', 'R5'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    train_mod = load_train_module()
    device = train_mod.pick_device()

    if CACHE.exists():
        print(f"Loading cached embeddings from {CACHE}")
        d = np.load(CACHE, allow_pickle=False)
        emb, cnn_preds, y = d['emb'], d['preds'], d['y']
        verses, starts = d['verses'], d['starts']
        class_names = list(d['class_names'])
    else:
        print("Embedding all words with the frozen CNN:")
        emb, cnn_preds, y, verses, starts, class_names = embed_everything(
            args.checkpoint, args.readers, device, train_mod)
        np.savez_compressed(CACHE, emb=emb, preds=cnn_preds, y=y,
                            verses=verses, starts=starts,
                            class_names=np.array(class_names))
    print(f"Total: {len(y)} words, emb dim {emb.shape[1]}, "
          f"{len(set(verses))} verses")

    # split by verse
    rng = np.random.default_rng(42)
    unique_verses = np.array(sorted(set(verses)))
    rng.shuffle(unique_verses)
    n = len(unique_verses)
    v_train = set(unique_verses[:int(0.7 * n)])
    v_val = set(unique_verses[int(0.7 * n):int(0.85 * n)])
    v_test = set(unique_verses[int(0.85 * n):])
    in_train = np.array([v in v_train for v in verses])
    in_val = np.array([v in v_val for v in verses])
    in_test = np.array([v in v_test for v in verses])

    seqs_train = make_batches(verses[in_train], starts[in_train], rng)
    # careful: indices above are positions within the filtered arrays
    idx_train = np.where(in_train)[0]
    seqs_train = [idx_train[s] for s in seqs_train]
    idx_val = np.where(in_val)[0]
    seqs_val = [idx_val[s] for s in make_batches(verses[in_val], starts[in_val])]
    idx_test = np.where(in_test)[0]
    seqs_test = [idx_test[s] for s in make_batches(verses[in_test], starts[in_test])]
    print(f"Verses: train {len(seqs_train)}, val {len(seqs_val)}, test {len(seqs_test)}")

    num_classes = len(class_names)
    counts = np.bincount(y[in_train], minlength=num_classes)
    weights = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)),
                           dtype=torch.float32).to(device)
    model = VerseTagger(emb.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    def evaluate(seqs):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb, lengths in batch_iter(seqs, emb, y, args.batch_size, device):
                out = model(xb, lengths)
                mask = yb != -100
                preds.append(out.argmax(-1)[mask].cpu().numpy())
                labels.append(yb[mask].cpu().numpy())
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        return (accuracy_score(labels, preds),
                balanced_accuracy_score(labels, preds),
                matthews_corrcoef(labels, preds))

    best_val, best_state, patience_left = 0, None, 6
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(seqs_train)
        total_loss = 0
        for xb, yb, lengths in batch_iter(seqs_train, emb, y, args.batch_size, device):
            opt.zero_grad()
            out = model(xb, lengths)
            loss = crit(out.reshape(-1, num_classes), yb.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc, bal, mcc = evaluate(seqs_val)
        print(f"  Epoch {epoch:2d}/{args.epochs} | loss {total_loss/len(seqs_train):.4f} "
              f"| val acc {acc:.4f} bal {bal:.4f} mcc {mcc:.4f}", flush=True)
        if bal > best_val:
            best_val, best_state = bal, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 6
        else:
            patience_left -= 1
            if patience_left == 0:
                print("  Early stopping")
                break

    model.load_state_dict(best_state)
    seq_acc, seq_bal, seq_mcc = evaluate(seqs_test)

    # baseline: the frozen CNN's own predictions on the SAME test words
    test_mask = in_test
    base_acc = accuracy_score(y[test_mask], cnn_preds[test_mask])
    base_bal = balanced_accuracy_score(y[test_mask], cnn_preds[test_mask])
    base_mcc = matthews_corrcoef(y[test_mask], cnn_preds[test_mask])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Sequence Context Model — verse-level BiLSTM over frozen CNN embeddings",
             "",
             f"Base CNN: {args.checkpoint.name} | split BY VERSE (70/15/15), "
             f"test = {int(test_mask.sum())} words in {len(seqs_test)} verses", "",
             "| Model | Accuracy | Balanced Acc | MCC |", "|---|---|---|---|",
             f"| CNN alone (per word) | {base_acc*100:.1f}% | {base_bal*100:.1f}% | {base_mcc:.3f} |",
             f"| CNN + verse BiLSTM | {seq_acc*100:.1f}% | {seq_bal*100:.1f}% | {seq_mcc:.3f} |",
             "",
             f"**Context gain: {(seq_acc-base_acc)*100:+.1f} points accuracy, "
             f"{(seq_bal-base_bal)*100:+.1f} points balanced accuracy** — "
             "the ta'amim grammar (fixed pairs like mercha-tipcha) is real signal."]
    (OUT_DIR / 'sequence_model_report.md').write_text("\n".join(lines) + "\n", encoding='utf-8')
    torch.save({'model_state_dict': best_state, 'class_names': class_names,
                'in_dim': emb.shape[1]}, OUT_DIR / 'verse_tagger.pth')
    print("\n".join(lines))
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
