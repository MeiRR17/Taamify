"""
Is the traditional hierarchy visible in the embedding space itself?
===================================================================
`analysis_hierarchy_transfer.py` asks whether the model's *mistakes* respect the
ta'amim ranks. This asks something stronger and independent of any threshold:
whether the geometry the network learned is organised by rank at all.

Two tests, both on the saved penultimate-layer embeddings, so nothing retrains:

1. **Rank coherence.** Are two ta'amim of the same traditional rank closer in
   embedding space than two of different ranks? Reported as the gap between the
   mean within-rank and between-rank centroid distance, with a permutation test
   that shuffles the rank labels — so a rank that happens to contain the two
   most common marks cannot produce the effect on its own.

2. **Silhouette by rank.** How well the ranks separate as clusters, compared to
   how well the ta'amim themselves do. A rank silhouette near the ta'am
   silhouette means the coarse structure carries most of the geometry.

Run per fold. The interesting comparison is the cross-tradition fold (R6)
against the within-tradition ones: if the geometry stays rank-organised where
the melody is foreign, the structure is not carried by the tune.

Usage:
  python analysis_embedding_rank.py --suffix _cap40000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_taam_similarity import rank_of  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results"
TRADITION = {"R1": "Sephardic-Jerusalem", "R2": "Sephardic-Jerusalem",
             "R6": "Ashkenazi"}


def centroids(emb, y, n_classes):
    out = np.full((n_classes, emb.shape[1]), np.nan, dtype=np.float64)
    for c in range(n_classes):
        m = y == c
        if m.sum() >= 5:
            out[c] = emb[m].mean(0)
    return out


def cosine_matrix(C):
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = C / norm
    return 1.0 - U @ U.T


def rank_gap(D, ranks, valid):
    """mean(between-rank distance) - mean(within-rank distance)."""
    idx = np.flatnonzero(valid)
    within, between = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            (within if ranks[i] == ranks[j] else between).append(D[i, j])
    if not within or not between:
        return None
    return float(np.mean(between) - np.mean(within)), len(within), len(between)


def permutation_p(D, ranks, valid, observed, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid)
    labels = np.array([ranks[i] for i in idx])
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        r = dict(zip(idx, perm))
        w, b = [], []
        for a in range(len(idx)):
            for c in range(a + 1, len(idx)):
                i, j = idx[a], idx[c]
                (w if r[i] == r[j] else b).append(D[i, j])
        if w and b and (np.mean(b) - np.mean(w)) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_cap40000")
    ap.add_argument("--model", default="deep_cnn")
    ap.add_argument("--folds", nargs="+", default=["R1", "R2", "R6"])
    ap.add_argument("--classes", type=int, default=11)
    ap.add_argument("--perms", type=int, default=3000)
    args = ap.parse_args()

    from sklearn.metrics import silhouette_score

    rows = []
    for holdout in args.folds:
        p = (RESULTS / f"loro_{holdout}_top{args.classes}{args.suffix}"
             / args.model / f"{args.model}_embeddings.npz")
        if not p.exists():
            print(f"  missing {p.relative_to(PROJECT_DIR)} — skipped")
            continue
        d = np.load(p, allow_pickle=True)
        emb, y = d["embeddings"].astype(np.float64), d["labels"]
        names = [str(c) for c in d["class_names"]]
        C = centroids(emb, y, len(names))
        valid = ~np.isnan(C).any(1)
        D = cosine_matrix(np.nan_to_num(C))
        ranks = [rank_of(n) for n in names]

        gap = rank_gap(D, ranks, valid)
        if gap is None:
            continue
        observed, n_w, n_b = gap
        pval = permutation_p(D, ranks, valid, observed, n_perm=args.perms)

        # Silhouettes on a subsample: full pairwise on 60k points is wasteful.
        rng = np.random.default_rng(0)
        take = rng.choice(len(emb), size=min(6000, len(emb)), replace=False)
        rank_ids = np.array([ranks[c] for c in y[take]])
        sil_rank = (silhouette_score(emb[take], rank_ids, metric="cosine")
                    if len(set(rank_ids)) > 1 else float("nan"))
        sil_taam = (silhouette_score(emb[take], y[take], metric="cosine")
                    if len(set(y[take])) > 1 else float("nan"))
        rows.append((holdout, observed, pval, n_w, n_b, sil_rank, sil_taam))

    if not rows:
        raise SystemExit("no folds found")

    print(f"\n{'fold':<6}{'tradition':<22}{'rank gap':>10}{'perm p':>9}"
          f"{'pairs w/b':>12}{'sil(rank)':>11}{'sil(ta_am)':>12}")
    print("-" * 84)
    for h, obs, p, nw, nb, sr, st in rows:
        print(f"{h:<6}{TRADITION.get(h, '?'):<22}{obs:>+10.4f}{p:>9.4f}"
              f"{f'{nw}/{nb}':>12}{sr:>11.3f}{st:>12.3f}")
    print("-" * 84)
    print("rank gap = mean between-rank centroid distance minus mean within-rank\n"
          "distance; positive means same-rank ta'amim sit closer together. perm p\n"
          "shuffles the rank labels, so an effect driven by which marks happen to\n"
          "share a rank does not count. sil(rank) vs sil(ta'am): if the coarse\n"
          "grouping separates nearly as well as the marks themselves, the geometry\n"
          "is organised by hierarchy, not just by class.")

    out = RESULTS / "v3_comparison" / f"embedding_rank{args.suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([{"fold": h, "tradition": TRADITION.get(h), "rank_gap": o,
                    "perm_p": p, "sil_rank": sr, "sil_taam": st}
                   for h, o, p, _, _, sr, st in rows], f, indent=1)
    print(f"\nwrote {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
