"""
Is the Ben13 "dialect" a dialect, or is it his word windows?
===========================================================
research_invariants.py found large reader disagreements concentrated in
`turns`, `range`, `rise_first` and `sd` -- and always with Ben13 as the
outlier. The duration table shows why that is suspicious: Ben13's median word
is 5.76 s for Zaqef Gadol against 1.5-1.7 s for R1/R2/R8, 4.9 s for Telisha
Gedola against 1.3-1.6 s, while for Mercha and Munach his words are SHORTER
than everyone's. Ben13's corpus is cut into phrases AT the disjunctive
accents, so a disjunctive is the last word before a cut and can absorb
whatever the cut left behind.

If that is what is happening, the reader disagreement should be confined to
features whose value grows with how much audio is inside the window
(`turns`, `range`, `sd`, `rise_first`) and absent from features computed on
the time-normalised 60-point curve, which are scale-free in time
(`peak_pos`, `trough_pos`, `net`, `end_vs_tonic`, `fall_last`).

Three tests:
  1. per feature, correlate the cross-reader bias with the Ben13/others
     duration ratio across ta'amim;
  2. measure where the voiced signal actually sits inside Ben13's windows for
     the ta'amim with the worst ratio -- trailing silence has a signature;
  3. redo the reader-agreement table on R1/R2/R8 ONLY. Those three share one
     alignment method (verse-level CTC into a parasha wav), so anything that
     still disagrees there is not a Ben13 artifact.

Usage:
  venv/bin/python Data_Prep/research_confounds.py
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
OUT = P / "results" / "melody_research" / "confounds.json"
TIME_FREE = ["peak_pos", "trough_pos", "net", "end_vs_tonic", "head_vs_tonic",
             "fall_last", "rise_first", "final_slope", "initial_slope",
             "peak_minus_stress"]
SCALE = ["turns", "range", "sd"]


def main():
    rows = json.loads(CURVES.read_text(encoding="utf-8"))
    inv = json.loads((P / "results" / "melody_research" /
                      "invariants.json").read_text(encoding="utf-8"))["real"]

    # ---- 1. bias vs duration ratio ----------------------------------------
    dur = defaultdict(lambda: defaultdict(list))
    for r in rows:
        dur[r["taam"]][r["reader"]].append(r["dur"])
    ratio = {}
    for t, d in dur.items():
        o = [np.median(d[k]) for k in ("R1", "R2", "R8") if k in d and len(d[k]) > 5]
        if "BEN13R" in d and len(d["BEN13R"]) > 5 and o:
            ratio[t] = float(np.median(d["BEN13R"]) / np.mean(o))

    print("Spearman(|cross-reader bias|, Ben13/others duration ratio) per feature")
    print(f"  {'feature':<20}{'rho':>8}{'n_taam':>8}")
    from scipy.stats import spearmanr
    corr = {}
    names = FEATURE_NAMES + ["peak_minus_stress"]
    for name in names:
        xs, ys = [], []
        for t, e in inv.items():
            f = e["features"].get(name)
            if f is None or t not in ratio:
                continue
            # bias against Ben13 specifically
            bb = [abs(v) for k, v in f["bias"].items() if "BEN13R" in k]
            if not bb:
                continue
            xs.append(max(bb) / max(f["rest_sd"], 1e-6))
            ys.append(ratio[t])
        if len(xs) >= 8:
            rho, p = spearmanr(xs, ys)
            corr[name] = [float(rho), float(p), len(xs)]
            tag = "  <- scale-with-window" if name in SCALE else ""
            print(f"  {name:<20}{rho:>8.2f}{len(xs):>8}  p={p:.3f}{tag}")

    # ---- 2. where the voiced signal sits in Ben13 windows ------------------
    print("\nBen13 window occupancy (voiced fraction, by ta'am duration ratio)")
    vf = defaultdict(list)
    for r in rows:
        if r["reader"] == "BEN13R":
            vf[r["taam"]].append(r["voiced_frac"])
    for t in sorted(ratio, key=lambda x: -ratio[x])[:8]:
        print(f"  {t:<18} ratio={ratio[t]:>5.2f}  median_dur="
              f"{np.median(dur[t]['BEN13R']):>5.2f}s  voiced="
              f"{np.mean(vf[t]):.3f}  n={len(vf[t])}")
    print("  (all-reader mean voiced fraction for reference: "
          f"{np.mean([r['voiced_frac'] for r in rows]):.3f})")

    # ---- 3. reader agreement without Ben13 --------------------------------
    print("\nagreement among R1/R2/R8 only (one alignment method, three voices)")
    sub = [r for r in rows if r["reader"] != "BEN13R"]
    by_rb = defaultdict(list)
    for r in sub:
        by_rb[(r["reader"], r["ref"].rsplit(" ", 1)[0])].append(r["med_hz"])
    tonic = {k: float(np.median(v)) for k, v in by_rb.items()}
    F, taam, reader = [], [], []
    for r in sub:
        off = 12 * np.log2(r["med_hz"] / tonic[(r["reader"], r["ref"].rsplit(" ", 1)[0])])
        f = features(r["curve"], r["dur"], r.get("stress"), off)
        F.append([f.get(n, np.nan) for n in names])
        taam.append(r["taam"])
        reader.append(r["reader"])
    F = np.array(F, float)
    taam = np.array(taam)
    reader = np.array(reader)

    out3 = {}
    print(f"  {'taam':<18}{'feature':<18}{'d_min':>7}{'spread/SD':>11}")
    for t in sorted(set(taam)):
        m = taam == t
        if m.sum() < 90:
            continue
        best = None
        for ci, name in enumerate(names):
            col = F[:, ci]
            ds, means = [], []
            ok = True
            for rd in ("R1", "R2", "R8"):
                v = col[m & (reader == rd)]
                w = col[(~m) & (reader == rd)]
                v, w = v[np.isfinite(v)], w[np.isfinite(w)]
                if len(v) < 25:
                    ok = False
                    break
                sp = np.sqrt((v.var() + w.var()) / 2)
                ds.append((v.mean() - w.mean()) / max(sp, 1e-9))
                means.append(v.mean())
            if not ok or len({np.sign(x) for x in ds}) != 1:
                continue
            restsd = np.nanstd(col)
            spread = (max(means) - min(means)) / max(restsd, 1e-9)
            score = min(abs(x) for x in ds)
            if best is None or score > best[0]:
                best = (score, name, spread, [round(float(x), 2) for x in means],
                        [round(float(x), 2) for x in ds])
        if best:
            out3[t] = {"feature": best[1], "d_min": best[0], "spread_over_sd": best[2],
                       "means": best[3], "d_rest": best[4]}
            print(f"  {t:<18}{best[1]:<18}{best[0]:>7.2f}{best[2]:>11.2f}"
                  f"   means={best[3]}")

    OUT.write_text(json.dumps({"dur_ratio": ratio, "bias_vs_dur": corr,
                               "r_only": out3}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n-> {OUT.relative_to(P)}")


if __name__ == "__main__":
    main()
