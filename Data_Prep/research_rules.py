"""
Turn the surviving features into rules, fitted on Ben13 and tested on the others.
================================================================================
This is the product's actual question. The model defines "correct" as "what
Ben13 does", so a candidate invariant must be CALIBRATED ON BEN13 ALONE and
then hold, unchanged, for three readers it never saw.

For every (ta'am, feature) the threshold is the value that best separates that
ta'am from all other ta'amim IN BEN13's DATA ONLY (max Youden J over the
feature's own grid). The threshold is then frozen and applied to R1, R2 and R8.

Reported per rule:
  hit_B13      fraction of Ben13's instances of the ta'am that satisfy it
  hit_min      the WORST of R1/R2/R8 -- a rule is only usable if it holds for
               the weakest voice, not on average
  base         the same rule's firing rate on every OTHER ta'am, averaged over
               readers: the rate a rule that says nothing would achieve
  lift         hit_min / base
  spread       max-min of the per-reader hit rates: how much the readers
               disagree about the rule they are all supposed to obey
  shuf_lift    the same quantity with ta'am labels permuted at slot level

A rule counts as a CROSS-READER INVARIANT when hit_min >= 0.70, lift >= 1.5,
spread <= 0.20, and the shuffled control shows lift <= 1.1.

Usage:
  venv/bin/python Data_Prep/research_rules.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from research_melody_lib import features, FEATURE_NAMES  # noqa: E402

CURVES = P / "results" / "melody_research" / "curves_matched.json"
OUT = P / "results" / "melody_research" / "rules.json"
OTHERS = ["R1", "R2", "R8"]
MIN_B13, MIN_OTHER = 40, 25
HIT_MIN, LIFT_MIN, SPREAD_MAX = 0.70, 1.5, 0.20
# features whose value grows with how much audio sits inside the window;
# research_confounds.py showed their cross-reader disagreement tracks the
# Ben13/others duration ratio at rho 0.59-0.80, so they are reported but never
# counted as invariants
WINDOW_SCALED = {"turns", "range", "sd", "dur", "rise_first"}


def build(rows):
    by_rb = defaultdict(list)
    for r in rows:
        by_rb[(r["reader"], r["ref"].rsplit(" ", 1)[0])].append(r["med_hz"])
    tonic = {k: float(np.median(v)) for k, v in by_rb.items()}
    names = FEATURE_NAMES + ["peak_minus_stress"]
    F = []
    for r in rows:
        off = 12 * np.log2(r["med_hz"] /
                           tonic[(r["reader"], r["ref"].rsplit(" ", 1)[0])])
        f = features(r["curve"], r["dur"], r.get("stress"), off)
        F.append([f.get(n, np.nan) for n in names])
    return np.array(F, float), names


def fit_rule(x_in, x_out):
    """Threshold + direction maximising Youden J on Ben13 only."""
    v = np.concatenate([x_in, x_out])
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return None
    grid = np.quantile(v, np.linspace(0.02, 0.98, 49))
    best = None
    for th in grid:
        for sgn in (1, -1):
            tpr = np.mean(sgn * x_in >= sgn * th)
            fpr = np.mean(sgn * x_out >= sgn * th)
            j = tpr - fpr
            if best is None or j > best[0]:
                best = (j, float(th), sgn, float(tpr), float(fpr))
    return best


def evaluate(F, names, taam, reader, tag):
    out = {}
    for t in sorted(set(taam)):
        m = taam == t
        if (m & (reader == "BEN13R")).sum() < MIN_B13:
            continue
        rules = {}
        for ci, name in enumerate(names):
            col = F[:, ci]
            xi = col[m & (reader == "BEN13R")]
            xo = col[(~m) & (reader == "BEN13R")]
            xi, xo = xi[np.isfinite(xi)], xo[np.isfinite(xo)]
            if len(xi) < MIN_B13 or len(xo) < 100:
                continue
            fit = fit_rule(xi, xo)
            if fit is None:
                continue
            j, th, sgn, tpr, fpr = fit
            hits, bases, ok = {"BEN13R": tpr}, {"BEN13R": fpr}, True
            for rd in OTHERS:
                a = col[m & (reader == rd)]
                b = col[(~m) & (reader == rd)]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) < MIN_OTHER or len(b) < 100:
                    ok = False
                    break
                hits[rd] = float(np.mean(sgn * a >= sgn * th))
                bases[rd] = float(np.mean(sgn * b >= sgn * th))
            if not ok:
                continue
            ho = [hits[r] for r in OTHERS]
            base = float(np.mean([bases[r] for r in OTHERS]))
            hit_min = float(min(ho))
            rules[name] = {
                "threshold": th, "direction": ">=" if sgn > 0 else "<=",
                "hit": {k: round(v, 3) for k, v in hits.items()},
                "base": round(base, 3), "hit_min": round(hit_min, 3),
                "lift": round(hit_min / base, 3) if base > 1e-6 else None,
                "spread": round(float(max(list(hits.values())) -
                                      min(list(hits.values()))), 3),
                "window_scaled": name in WINDOW_SCALED,
                "n_b13": int(len(xi)),
            }
        out[t] = rules
    return out


def main():
    rows = json.loads(CURVES.read_text(encoding="utf-8"))
    F, names = build(rows)
    taam = np.array([r["taam"] for r in rows])
    reader = np.array([r["reader"] for r in rows])
    slot = np.array([r["slot_id"] for r in rows])

    real = evaluate(F, names, taam, reader, "real")

    slots_u = np.unique(slot)
    first = {s: taam[np.where(slot == s)[0][0]] for s in slots_u}
    rng = np.random.default_rng(0)
    fake = dict(zip(slots_u, rng.permutation([first[s] for s in slots_u])))
    shuf = evaluate(F, names, np.array([fake[s] for s in slot]), reader, "shuf")

    verdict = {}
    print(f"{'taam':<18}{'rule':<40}{'B13':>6}{'min':>6}{'base':>6}"
          f"{'lift':>6}{'spr':>6}{'shufL':>7}")
    for t, rules in sorted(real.items()):
        passing = []
        for name, r in rules.items():
            if r["window_scaled"] or r["lift"] is None:
                continue
            if (r["hit_min"] >= HIT_MIN and r["lift"] >= LIFT_MIN
                    and r["spread"] <= SPREAD_MAX):
                sl = shuf.get(t, {}).get(name, {}).get("lift")
                if sl is not None and sl > 1.1:
                    continue
                passing.append((r["lift"], name, r, sl))
        passing.sort(reverse=True)
        verdict[t] = {"n_rules": len(passing),
                      "rules": [{"feature": n, **r} for _, n, r, _ in passing]}
        if not passing:
            print(f"{t:<18}{'-- no rule passes --':<40}")
            continue
        for lift, name, r, sl in passing[:3]:
            txt = f"{name} {r['direction']} {r['threshold']:.2f}"
            print(f"{t:<18}{txt:<40}{r['hit']['BEN13R']:>6.2f}{r['hit_min']:>6.2f}"
                  f"{r['base']:>6.2f}{lift:>6.2f}{r['spread']:>6.2f}"
                  f"{(sl if sl is not None else float('nan')):>7.2f}")

    OUT.write_text(json.dumps({"real": real, "shuffled": shuf,
                               "verdict": verdict}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n-> {OUT.relative_to(P)}")


if __name__ == "__main__":
    main()
