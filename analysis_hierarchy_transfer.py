"""
Does the traditional hierarchy survive when the melody does not?
================================================================
The LORO folds split cleanly by musical tradition. R1 and R2 are both Sephardic
Jerusalem, so holding out either still leaves the other in training. R6 is the
Ashkenazi PocketTorah reader, and holding it out is the only fold where the
model is tested on a tradition it has never heard — where accuracy collapses
from ~46-51% to 16.5%.

That collapse sets up the question the project actually cares about. The
headline finding is that errors stay inside the same *traditional rank* of the
ta'amim hierarchy more often than chance — a network trained on audio alone
recovering the taxonomy the Masoretes encoded. If that holds even in the
cross-tradition fold, where the melody is foreign and the model can barely name
the mark, then the melodic surface is tradition-specific while the syntactic
structure underneath is shared. A model that fails to identify the ta'am but
still fails *towards the right rank* has learned something deeper than a tune.

Predictions come from the saved logits, so nothing is retrained.

The chance baseline is not 1/n: it respects how often each class is wrongly
predicted overall, so a model that simply over-predicts one crowded rank cannot
manufacture the effect.

Usage:
  python analysis_hierarchy_transfer.py --suffix _cap40000
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_taam_similarity import RANK_ORDER, rank_of  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results"

# Which tradition each reader sings in. R1/R2 come from the same Sephardic
# Jerusalem channels; R6 is PocketTorah's Ashkenazi reading.
TRADITION = {"R1": "Sephardic-Jerusalem", "R2": "Sephardic-Jerusalem",
             "R6": "Ashkenazi", "R7": "Yemenite"}


def rank_error_stats(y_true, y_pred, class_names):
    """Share of errors staying inside one rank, against a distribution-aware chance."""
    errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    if not errors:
        return None
    n = len(class_names)
    within = sum(1 for t, p in errors
                 if rank_of(class_names[t]) == rank_of(class_names[p]))
    observed = within / len(errors)

    pred_counts = Counter(p for _, p in errors)
    total = sum(pred_counts.values())
    q = {c: pred_counts.get(c, 0) / total for c in range(n)}
    chance = 0.0
    true_counts = Counter(t for t, _ in errors)
    for t, n_t in true_counts.items():
        denom = sum(q[j] for j in range(n) if j != t)
        same = sum(q[j] for j in range(n) if j != t
                   and rank_of(class_names[j]) == rank_of(class_names[t]))
        chance += (n_t / len(errors)) * (same / denom if denom else 0.0)
    return observed, chance, len(errors)


def bootstrap_ci(y_true, y_pred, class_names, n_boot=400, seed=0):
    """Percentile CI on the observed-minus-chance lift, resampling errors."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(y_true != y_pred)
    if idx.size < 30:
        return None
    lifts = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=idx.size, replace=True)
        st = rank_error_stats(y_true[take], y_pred[take], class_names)
        if st:
            lifts.append(st[0] - st[1])
    if not lifts:
        return None
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def load_fold(path: Path):
    d = np.load(path, allow_pickle=True)
    y = d["labels"]
    pred = d["logits"].argmax(1)
    names = [str(c) for c in d["class_names"]]
    return y, pred, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_cap40000",
                    help="results dir suffix, e.g. _cap40000 for the size-matched run")
    ap.add_argument("--model", default="deep_cnn")
    ap.add_argument("--folds", nargs="+", default=["R1", "R2", "R6"])
    ap.add_argument("--classes", type=int, default=11)
    args = ap.parse_args()

    rows = []
    for holdout in args.folds:
        p = (RESULTS / f"loro_{holdout}_top{args.classes}{args.suffix}"
             / args.model / f"{args.model}_embeddings.npz")
        if not p.exists():
            print(f"  missing {p.relative_to(PROJECT_DIR)} — skipped")
            continue
        y, pred, names = load_fold(p)
        acc = float((y == pred).mean())
        st = rank_error_stats(y, pred, names)
        ci = bootstrap_ci(y, pred, names)
        rows.append((holdout, acc, st, ci, len(y), names))

    if not rows:
        raise SystemExit("no folds found — run the LORO experiment first")

    print(f"\n{'fold':<6}{'tradition':<22}{'n':>8}{'acc':>8}"
          f"{'errors':>9}{'within-rank':>13}{'chance':>9}{'lift':>9}{'95% CI':>18}")
    print("-" * 104)
    for holdout, acc, st, ci, n, _ in rows:
        obs, ch, n_err = st
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—"
        print(f"{holdout:<6}{TRADITION.get(holdout, '?'):<22}{n:>8,}{acc:>8.3f}"
              f"{n_err:>9,}{obs:>12.1%}{ch:>9.1%}{obs - ch:>+9.1%}{ci_s:>18}")
    print("-" * 104)
    print("within-rank = share of the model's mistakes that land on a ta'am of the\n"
          "same traditional rank. chance respects the model's own prediction mix, so\n"
          "over-predicting a crowded rank cannot fake it. lift = observed - chance;\n"
          "a CI excluding zero means the hierarchy is showing through the errors.")

    # Any reader outside the Sephardic-Jerusalem pair (R1/R2) is a cross-
    # tradition fold — Ashkenazi (R6) and Yemenite (R7) alike.
    cross = [r for r in rows if TRADITION.get(r[0]) not in
             (None, "Sephardic-Jerusalem")]
    within = [r for r in rows if TRADITION.get(r[0]) == "Sephardic-Jerusalem"]
    if cross and within:
        cl = np.mean([r[2][0] - r[2][1] for r in cross])
        wl = np.mean([r[2][0] - r[2][1] for r in within])
        ca = np.mean([r[1] for r in cross])
        wa = np.mean([r[1] for r in within])
        print(f"\nwithin-tradition folds : accuracy {wa:.3f}, rank lift {wl:+.1%}")
        print(f"cross-tradition fold   : accuracy {ca:.3f}, rank lift {cl:+.1%}")
        print(f"\naccuracy retained across traditions: {100 * ca / max(wa, 1e-9):.0f}%")
        if cl > 0:
            print(f"rank structure retained          : {100 * cl / max(wl, 1e-9):.0f}%")
            print("\nIf the rank lift survives far better than the accuracy does, the\n"
                  "melody is what fails to transfer, not the structure it encodes.")
        else:
            print("\nThe rank effect does not survive the tradition change: what the model\n"
                  "learned about the hierarchy was carried by the melody it knew.")

    # Per-rank breakdown for the cross-tradition fold, where it matters most.
    for holdout, acc, st, ci, n, names in rows:
        if TRADITION.get(holdout) not in ("Ashkenazi", "Yemenite"):
            continue
        p = (RESULTS / f"loro_{holdout}_top{args.classes}{args.suffix}"
             / args.model / f"{args.model}_embeddings.npz")
        y, pred, names = load_fold(p)
        print(f"\ncross-tradition fold ({holdout}) — where the errors land, by true rank:")
        print(f"{'true rank':<12}{'errors':>9}{'same rank':>12}{'share':>9}"
              f"{'chance':>9}{'lift':>9}")
        print("-" * 62)
        # Chance has to be computed per rank, not taken from the global figure:
        # the ranks differ wildly in how many marks they hold and how often the
        # model predicts into them, so a single baseline would flatter the
        # crowded ranks and punish the sparse ones.
        err_all = y != pred
        pred_counts = Counter(pred[err_all].tolist())
        total = sum(pred_counts.values())
        q = {c: pred_counts.get(c, 0) / total for c in range(len(names))}
        for r in RANK_ORDER:
            mask = np.array([rank_of(names[t]) == r for t in y]) & err_all
            if mask.sum() < 20:
                continue
            same = sum(1 for t, pp in zip(y[mask], pred[mask])
                       if rank_of(names[pp]) == r)
            ch = 0.0
            for t, n_t in Counter(y[mask].tolist()).items():
                denom = sum(q[j] for j in range(len(names)) if j != t)
                same_q = sum(q[j] for j in range(len(names))
                             if j != t and rank_of(names[j]) == r)
                ch += (n_t / mask.sum()) * (same_q / denom if denom else 0.0)
            obs_r = same / mask.sum()
            print(f"{r:<12}{mask.sum():>9,}{same:>12,}{obs_r:>9.1%}"
                  f"{ch:>9.1%}{obs_r - ch:>+9.1%}")
        print("-" * 62)
        print("A rank whose lift is near zero contributes nothing to the aggregate\n"
              "effect, however large its raw share looks: a crowded rank catches\n"
              "stray predictions by size alone.")

    out = RESULTS / "v3_comparison" / f"hierarchy_transfer{args.suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([{"fold": h, "tradition": TRADITION.get(h), "accuracy": a,
                    "within_rank": s[0], "chance": s[1], "lift": s[0] - s[1],
                    "n_errors": s[2], "ci": c}
                   for h, a, s, c, _, _ in rows], f, indent=1)
    print(f"\nwrote {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
