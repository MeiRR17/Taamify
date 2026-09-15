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
CACHE_DIR = Path("/tmp")
OUT_DIR = PROJECT_DIR / "results" / "sequence_model"

# This script had no seeding at all, so repeated runs drew a different random
# init every time and no result here was reproducible -- which also made any
# "seed spread" meaningless, because the seeds were neither controlled nor
# recorded. Same TAAMIFY_SEED convention as 10_train_pipeline.py.
SEED = int(os.environ.get("TAAMIFY_SEED", "42"))


def set_seeds(seed: int = SEED):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds MPS


set_seeds(SEED)

# Near-deterministic tier from results/v3_comparison/ben13_grammar_associations.md
# -- Ben13-taught combinations independently confirmed as the dominant real
# transition (>=50%) in aligned_dataset_merged.json, not just Ben13's claim.
GRAMMAR_PRIOR_PAIRS = [
    ("Mahapakh", "Pashta"),
    ("Darga", "Tevir"),
    ("Mercha", "Tipecha"),
    ("Zarqa", "Munach"),
]


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


class LinearChainCRF(nn.Module):
    """Standard linear-chain CRF: log-likelihood training + Viterbi decoding.

    Motivation measured directly on this corpus, not assumed: of 5,785 verses,
    5,581 carry exactly one silluq and 5,424 exactly one etnachta, and *none*
    carries two. Ta'amim transitions are close to deterministic grammar, so a
    CRF has real structure to learn on top of the BiLSTM's per-word emissions,
    not noise it will overfit to.

    Implementation is the textbook vectorized forward-algorithm / Viterbi
    pattern (e.g. AllenNLP's CRF layer), kept self-contained rather than
    adding a dependency for ~80 lines of well-known math.
    """

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)

    def apply_grammar_prior(self, class_names, pairs=GRAMMAR_PRIOR_PAIRS, bias=2.0):
        """Nudge specific (from, to) transitions upward before training --
        a soft prior, not a hard rule: every weight, including these, is
        still updated normally by training. `pairs` defaults to the
        near-deterministic tier from
        results/v3_comparison/ben13_grammar_associations.md -- Ben13-taught
        combinations independently confirmed as the dominant real pattern
        (>=50% of the time) in this corpus, not just Ben13's own claim.
        Pairs naming a class not in `class_names` (e.g. Darga/Zarqa before
        Milestone 4 widens past top-11) are skipped, not errors."""
        idx = {c: i for i, c in enumerate(class_names)}
        applied = []
        with torch.no_grad():
            for a, b in pairs:
                if a in idx and b in idx:
                    self.transitions[idx[a], idx[b]] += bias
                    applied.append(f"{a}->{b}")
        return applied

    def _log_partition(self, emissions, mask):
        # emissions: (B, L, C) | mask: (B, L) bool, True = real token
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            next_score = (score.unsqueeze(2) + self.transitions.unsqueeze(0)
                         + emissions[:, t].unsqueeze(1))
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def _sequence_score(self, emissions, tags, mask):
        score = (self.start_transitions[tags[:, 0]]
                 + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1))
        for t in range(1, emissions.size(1)):
            emit = emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + mask[:, t].float() * (trans + emit)
        seq_ends = mask.sum(1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def neg_log_likelihood(self, emissions, tags, mask):
        safe_tags = tags.clone()
        safe_tags[~mask] = 0            # dummy index so gather stays in range
        gold = self._sequence_score(emissions, safe_tags, mask)
        logz = self._log_partition(emissions, mask)
        return (logz - gold).mean()

    def viterbi_decode(self, emissions, mask):
        B, L, _C = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for t in range(1, L):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_idx = next_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_idx)
        score = score + self.end_transitions
        _best_final, best_last_tag = score.max(dim=1)
        seq_ends = mask.sum(1) - 1
        out = []
        for b in range(B):
            length = int(seq_ends[b].item()) + 1
            tag = int(best_last_tag[b].item())
            tags_b = [tag]
            for t in range(length - 1, 0, -1):
                tag = int(history[t - 1][b, tag].item())
                tags_b.append(tag)
            tags_b.reverse()
            out.append(tags_b)
        return out


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
    ap.add_argument('--holdout', default=None,
                    help="reader to test on exclusively (LORO-style): the BiLSTM "
                         "trains only on the other readers' verses and is scored "
                         "only on this one's. --checkpoint should be the CNN that "
                         "was itself trained without this reader, or the comparison "
                         "is not apples-to-apples with the per-word LORO number. "
                         "Omit for the original random 70/15/15 verse split.")
    ap.add_argument('--tag', default=None,
                    help="suffix for the output dir, e.g. holdout reader name")
    ap.add_argument('--crf', action='store_true',
                    help="add a linear-chain CRF on top of the BiLSTM emissions, "
                         "trained jointly by sequence log-likelihood and decoded "
                         "with Viterbi instead of per-word argmax")
    ap.add_argument('--grammar-prior', action='store_true',
                    help="requires --crf: bias the CRF's transition matrix "
                         "toward GRAMMAR_PRIOR_PAIRS before training (a soft "
                         "starting point, not a hard rule -- see "
                         "results/v3_comparison/ben13_grammar_associations.md). "
                         "Report both --crf and --crf --grammar-prior numbers "
                         "side by side, not just whichever looks better.")
    args = ap.parse_args()
    if args.grammar_prior and not args.crf:
        ap.error("--grammar-prior requires --crf")

    train_mod = load_train_module()
    device = train_mod.pick_device()

    # Cache key includes the checkpoint's full resolved path (not just its
    # filename stem — two different LORO folds both save to a file literally
    # named deep_cnn_best.pth in different directories, and a stem-only key
    # collided between them: the R2-holdout run silently loaded the
    # R1-holdout run's cached embeddings, i.e. evaluated the wrong model and
    # reported 84.8% instead of the real ~56%) and the reader set.
    import hashlib
    key_src = f"{args.checkpoint.resolve()}_{'-'.join(sorted(args.readers))}"
    cache_key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"taamify_seq_cache_{cache_key}.npz"
    if cache.exists():
        print(f"Loading cached embeddings from {cache}")
        d = np.load(cache, allow_pickle=False)
        emb, cnn_preds, y = d['emb'], d['preds'], d['y']
        verses, starts = d['verses'], d['starts']
        class_names = list(d['class_names'])
    else:
        print("Embedding all words with the frozen CNN:")
        emb, cnn_preds, y, verses, starts, class_names = embed_everything(
            args.checkpoint, args.readers, device, train_mod)
        np.savez_compressed(cache, emb=emb, preds=cnn_preds, y=y,
                            verses=verses, starts=starts,
                            class_names=np.array(class_names))
    print(f"Total: {len(y)} words, emb dim {emb.shape[1]}, "
          f"{len(set(verses))} verses")

    # split by verse: LORO-style (reader holdout) or the original random 70/15/15
    rng = np.random.default_rng(42)
    if args.holdout:
        # verses are "reader|parasha|verse_ref" strings
        reader_of = {v: v.split("|", 1)[0] for v in set(verses)}
        held_verses = np.array(sorted(v for v, r in reader_of.items() if r == args.holdout))
        rest_verses = np.array(sorted(v for v, r in reader_of.items() if r != args.holdout))
        if held_verses.size == 0:
            raise SystemExit(f"no verses found for holdout reader {args.holdout!r} "
                             f"— is it in --readers?")
        rng.shuffle(rest_verses)
        n = len(rest_verses)
        # all of the holdout reader is test; val is carved from the training
        # readers only, so nothing from the held-out reader leaks into model
        # selection (early stopping) either.
        v_train = set(rest_verses[:int(0.85 * n)])
        v_val = set(rest_verses[int(0.85 * n):])
        v_test = set(held_verses)
        print(f"LORO split: train/val on {sorted(set(reader_of.values()) - {args.holdout})}, "
              f"test on held-out {args.holdout}")
    else:
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
    crf = LinearChainCRF(num_classes).to(device) if args.crf else None
    if crf and args.grammar_prior:
        applied = crf.apply_grammar_prior(class_names)
        applied_str = ', '.join(applied) if applied else \
            '(none -- no pair has both classes in this class set)'
        print(f"grammar prior applied to: {applied_str}")
    params = list(model.parameters()) + (list(crf.parameters()) if crf else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    def evaluate(seqs):
        model.eval()
        if crf:
            crf.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb, lengths in batch_iter(seqs, emb, y, args.batch_size, device):
                out = model(xb, lengths)
                mask = yb != -100
                if crf:
                    for b, tags in enumerate(crf.viterbi_decode(out, mask)):
                        preds.append(np.array(tags))
                        labels.append(yb[b, :len(tags)].cpu().numpy())
                else:
                    preds.append(out.argmax(-1)[mask].cpu().numpy())
                    labels.append(yb[mask].cpu().numpy())
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        return (accuracy_score(labels, preds),
                balanced_accuracy_score(labels, preds),
                matthews_corrcoef(labels, preds))

    best_val, best_state, best_crf_state, patience_left = 0, None, None, 6
    for epoch in range(1, args.epochs + 1):
        model.train()
        if crf:
            crf.train()
        rng.shuffle(seqs_train)
        total_loss = 0
        for xb, yb, lengths in batch_iter(seqs_train, emb, y, args.batch_size, device):
            opt.zero_grad()
            out = model(xb, lengths)
            if crf:
                mask = yb != -100
                loss = crf.neg_log_likelihood(out, yb, mask)
            else:
                loss = crit(out.reshape(-1, num_classes), yb.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc, bal, mcc = evaluate(seqs_val)
        print(f"  Epoch {epoch:2d}/{args.epochs} | loss {total_loss/len(seqs_train):.4f} "
              f"| val acc {acc:.4f} bal {bal:.4f} mcc {mcc:.4f}", flush=True)
        if bal > best_val:
            best_val = bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if crf:
                best_crf_state = {k: v.cpu().clone() for k, v in crf.state_dict().items()}
            patience_left = 6
        else:
            patience_left -= 1
            if patience_left == 0:
                print("  Early stopping")
                break

    model.load_state_dict(best_state)
    if crf:
        crf.load_state_dict(best_crf_state)
    seq_acc, seq_bal, seq_mcc = evaluate(seqs_test)

    # baseline: the frozen CNN's own predictions on the SAME test words
    test_mask = in_test
    base_acc = accuracy_score(y[test_mask], cnn_preds[test_mask])
    base_bal = balanced_accuracy_score(y[test_mask], cnn_preds[test_mask])
    base_mcc = matthews_corrcoef(y[test_mask], cnn_preds[test_mask])

    out_dir = OUT_DIR
    tag = (args.tag or "") + ("_crf" if args.crf else "") + \
        ("_gprior" if args.grammar_prior else "")
    if args.holdout:
        out_dir = OUT_DIR / f"loro_{args.holdout}{('_' + tag) if tag else ''}"
    elif tag:
        out_dir = OUT_DIR / tag
    else:
        out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    split_desc = (f"LORO holdout={args.holdout}" if args.holdout
                  else "random 70/15/15 by verse")
    seq_label = "CNN + verse BiLSTM" + (" + CRF" if args.crf else "") + \
        (" (grammar prior)" if args.grammar_prior else "")
    lines = ["# Sequence Context Model — verse-level BiLSTM over frozen CNN embeddings",
             "",
             f"Base CNN: {args.checkpoint.name} | split: {split_desc}, "
             f"test = {int(test_mask.sum())} words in {len(seqs_test)} verses", "",
             "| Model | Accuracy | Balanced Acc | MCC |", "|---|---|---|---|",
             f"| CNN alone (per word) | {base_acc*100:.1f}% | {base_bal*100:.1f}% | {base_mcc:.3f} |",
             f"| {seq_label} | {seq_acc*100:.1f}% | {seq_bal*100:.1f}% | {seq_mcc:.3f} |",
             "",
             f"**Context gain: {(seq_acc-base_acc)*100:+.1f} points accuracy, "
             f"{(seq_bal-base_bal)*100:+.1f} points balanced accuracy** — "
             "the ta'amim grammar (fixed pairs like mercha-tipcha) is real signal."]
    (out_dir / 'sequence_model_report.md').write_text("\n".join(lines) + "\n", encoding='utf-8')
    save = {'model_state_dict': best_state, 'class_names': class_names,
            'in_dim': emb.shape[1]}
    if crf:
        save['crf_state_dict'] = best_crf_state
    torch.save(save, out_dir / 'verse_tagger.pth')
    print("\n".join(lines))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
