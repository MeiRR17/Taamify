"""
Consolidate every measurement into one per-ta'am verdict.
=========================================================
Inputs, all produced by the scripts named in results/melody_research/REPORT.md:
  wordtest.json   text-controlled DTW discrimination, within- and cross-reader
  rules.json      Ben13-fitted thresholds tested on R1/R2/R8
  variance.json   identity index against a duration-matched null
  practical.json  leave-one-reader-out transfer of each scoring method

Verdict rules, stated before the table so they cannot be fitted to it:
  INVARIANT       cross-reader AUC >= 0.60 (well clear of the 0.50 shuffled
                  control) OR a Tier-A rule exists (Ben13-fitted threshold
                  held by all three other readers at >=0.70 with lift >=1.5
                  and spread <=0.20)
  WEAK            cross-reader AUC in [0.55, 0.60), or a Tier-B rule only
  NO INVARIANT    cross-reader AUC < 0.55 and no rule survives
  INSUFFICIENT    fewer than 25 usable pairs in any cell, or fewer than three
                  readers with >=25 instances

Usage:
  venv/bin/python Data_Prep/research_verdict.py
"""
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
MR = P / "results" / "melody_research"
WINDOW_SCALED = {"turns", "range", "sd", "dur", "rise_first"}


def tier_b(rules, t):
    out = []
    for name, r in rules["real"].get(t, {}).items():
        if name in WINDOW_SCALED or r["lift"] is None:
            continue
        sl = rules["shuffled"].get(t, {}).get(name, {}).get("lift")
        if sl is not None and sl > 1.1:
            continue
        if r["hit_min"] >= 0.60 and r["lift"] >= 1.2 and r["spread"] <= 0.30:
            out.append((r["lift"], name, r))
    out.sort(reverse=True)
    return out


def main():
    wt = json.loads((MR / "wordtest.json").read_text())["real"]
    rules = json.loads((MR / "rules.json").read_text())
    var = json.loads((MR / "variance.json").read_text())["real"]

    rowsf = []
    taams = sorted(set(list(wt) + list(rules["real"]) + list(var)))
    for t in taams:
        w = wt.get(t)
        a_in = w["auc_within"] if w else float("nan")
        a_cr = w["auc_cross"] if w else float("nan")
        a_crd = w["auc_cross_durmatched"] if w else float("nan")
        npairs = min(w["n_pairs"].values()) if w else 0
        tA = rules["verdict"].get(t, {}).get("rules", [])
        tB = tier_b(rules, t)
        ident = var.get(t, {}).get("identity", float("nan"))

        if (w is None or npairs < 25) and not tA:
            verdict = "INSUFFICIENT"
        elif (np.isfinite(a_cr) and a_cr >= 0.60) or tA:
            verdict = "INVARIANT"
        elif (np.isfinite(a_cr) and a_cr >= 0.55) or tB:
            verdict = "WEAK"
        else:
            verdict = "NO INVARIANT"

        best = None
        if tA:
            best = tA[0]
        elif tB:
            best = {"feature": tB[0][1], **tB[0][2]}
        rowsf.append({"taam": t, "auc_within": a_in, "auc_cross": a_cr,
                      "auc_cross_dm": a_crd, "identity": ident,
                      "n_pairs": npairs, "tierA": len(tA), "tierB": len(tB),
                      "verdict": verdict,
                      "rule": (f"{best['feature']} {best['direction']} "
                               f"{best['threshold']:.2f}" if best else ""),
                      "rule_hits": best["hit"] if best else None,
                      "rule_base": best["base"] if best else None})

    order = {"INVARIANT": 0, "WEAK": 1, "NO INVARIANT": 2, "INSUFFICIENT": 3}
    rowsf.sort(key=lambda r: (order[r["verdict"]], -(r["auc_cross"]
                                                     if np.isfinite(r["auc_cross"]) else 0)))
    print(f"{'taam':<17}{'verdict':<14}{'within':>8}{'cross':>8}{'cross_dm':>10}"
          f"{'ident':>8}  rule (fitted on Ben13)")
    for r in rowsf:
        print(f"{r['taam']:<17}{r['verdict']:<14}{r['auc_within']:>8.3f}"
              f"{r['auc_cross']:>8.3f}{r['auc_cross_dm']:>10.3f}"
              f"{r['identity']:>8.3f}  {r['rule']}")
    (MR / "verdict.json").write_text(json.dumps(rowsf, ensure_ascii=False, indent=1))
    print(f"\n-> {(MR / 'verdict.json').relative_to(P)}")


if __name__ == "__main__":
    main()
