"""
Is the tradition gap symmetric?
===============================
The cross-tradition collapse was measured one way round: train mostly on
Sephardic-Jerusalem, fail on Ashkenazi. That is compatible with a duller
explanation — R6 is simply harder audio, or its gold boundaries make it a
stricter test — so the claim needs the mirror.

Four transfers, each trained on a single reader and tested on a single reader,
all capped to the same 30,000 training samples so nothing turns on corpus size:

  cross-tradition:  train R6 (Ashkenazi)          -> test R1, test R2
  within-tradition: train R1 / R2 (Sephardic-Jm)  -> test R2, test R1

If both directions collapse while the within-tradition transfers hold, the gap
belongs to the tradition boundary and not to either reader. If only one
direction collapses, it belongs to that reader, and the tradition claim has to
be withdrawn.

The rank lift is reported alongside accuracy: the interesting case is a transfer
where accuracy dies but the hierarchy survives.

Usage:
  python analysis_symmetry.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_hierarchy_transfer import rank_error_stats, bootstrap_ci, load_fold

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results"

TRANSFERS = [
    ("trainR6_testR1", "R6 (Ashkenazi)", "R1 (Sephardic-Jm)", "cross"),
    ("trainR6_testR2", "R6 (Ashkenazi)", "R2 (Sephardic-Jm)", "cross"),
    ("trainR1_testR2", "R1 (Sephardic-Jm)", "R2 (Sephardic-Jm)", "within"),
    ("trainR2_testR1", "R2 (Sephardic-Jm)", "R1 (Sephardic-Jm)", "within"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deep_cnn")
    args = ap.parse_args()

    rows = []
    for tag, src, dst, kind in TRANSFERS:
        p = RESULTS / f"sym_{tag}" / args.model / f"{args.model}_embeddings.npz"
        if not p.exists():
            print(f"  missing {p.relative_to(PROJECT_DIR)} — skipped")
            continue
        y, pred, names = load_fold(p)
        acc = float((y == pred).mean())
        st = rank_error_stats(y, pred, names)
        ci = bootstrap_ci(y, pred, names)
        rows.append((tag, src, dst, kind, acc, st, ci, len(y)))

    if not rows:
        raise SystemExit("no symmetry runs found")

    print(f"\n{'train on':<20}{'test on':<20}{'kind':<8}{'n':>8}{'acc':>8}"
          f"{'rank lift':>11}{'95% CI':>18}")
    print("-" * 93)
    for tag, src, dst, kind, acc, st, ci, n in rows:
        lift = st[0] - st[1]
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—"
        print(f"{src:<20}{dst:<20}{kind:<8}{n:>8,}{acc:>8.3f}{lift:>+11.1%}{ci_s:>18}")
    print("-" * 93)

    cross = [r for r in rows if r[3] == "cross"]
    within = [r for r in rows if r[3] == "within"]
    if cross and within:
        ca, wa = np.mean([r[4] for r in cross]), np.mean([r[4] for r in within])
        cl = np.mean([r[5][0] - r[5][1] for r in cross])
        wl = np.mean([r[5][0] - r[5][1] for r in within])
        print(f"\nwithin-tradition : accuracy {wa:.3f}, rank lift {wl:+.1%}")
        print(f"cross-tradition  : accuracy {ca:.3f}, rank lift {cl:+.1%}")
        print(f"\naccuracy retained across the tradition boundary: "
              f"{100 * ca / max(wa, 1e-9):.0f}%")
        print(f"rank structure retained                       : "
              f"{100 * cl / max(wl, 1e-9):.0f}%")
        if len(cross) == 2:
            spread = abs(cross[0][4] - cross[1][4])
            print(f"\nthe two cross-tradition directions differ by {spread:.3f} in "
                  f"accuracy —\nif that is small, the gap is the boundary itself and "
                  f"not one awkward reader.")

    out = RESULTS / "v3_comparison" / "symmetry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([{"train": s, "test": d, "kind": k, "accuracy": a,
                    "within_rank": st[0], "chance": st[1], "lift": st[0] - st[1],
                    "ci": ci, "n": n}
                   for _, s, d, k, a, st, ci, n in rows], f, indent=1)
    print(f"\nwrote {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
