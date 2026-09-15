"""
Why is R1 different? A measurement, not a theory.
===================================================
R1 has sat at ~41-47% LORO across every configuration ever tried, while R2
and R8 reach 66-83% on the identical pipeline. Milestone-era work ruled out
data scarcity (4x training data buys 3.8 points) and text coverage (R1 has
the WIDEST coverage of the three). Every remaining explanation has been a
hand-wave about "something in R1's voice or reading style."

This measures the concrete, checkable properties of R1's corpus rows against
R2's and R8's, so the next explanation is at least anchored to a number:

  1. word duration distribution, overall and per ta'am
  2. CTC alignment confidence (ctc_score), where recorded
  3. alignment method mix (verse_ctc vs stored fallback)
  4. per-class balance -- is R1's class distribution itself skewed?
  5. duration SEPARABILITY between classes: if R1's classes overlap more in
     duration than R2's/R8's do, the acoustic task itself is harder on R1,
     independent of any model choice.

(5) is the interesting one: it needs no model at all, so it can't be blamed
on training. Reported as a per-reader effect size (between-class variance
over within-class variance -- a plain one-way F-statistic on log duration).

Usage:
  python Data_Prep/diagnose_r1_wall.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
META_DIR = PROJECT_DIR / "features_v3_word"
OUT = PROJECT_DIR / "results" / "r1_wall" / "diagnosis.txt"
READERS = ["R1", "R2", "R8"]
TOP11 = ['Tipecha', 'Munach', 'Mercha', 'Zaqef_Qatan', 'Etnachta', 'Pashta',
         'Siluk', 'Mahapakh', 'Tevir', 'Revia', 'Qadma']

L = []
def say(s=""):
    L.append(s)
    print(s, flush=True)


def f_statistic(groups):
    """One-way F: between-group variance / within-group variance."""
    groups = [np.asarray(g) for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    k, n = len(groups), len(all_vals)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    if ss_within <= 0 or n - k <= 0:
        return float("nan")
    return (ss_between / (k - 1)) / (ss_within / (n - k))


def main():
    with open(DATASET, encoding="utf-8") as f:
        records = json.load(f)
    by_reader = defaultdict(list)
    for r in records:
        if r["reader"] in READERS:
            by_reader[r["reader"]].append(r)

    say("=" * 72)
    say("  R1 wall diagnosis — corpus properties, no model involved")
    say("=" * 72)

    say("\n## 1. word duration (seconds)")
    say(f"  {'reader':<8}{'n':>8}{'median':>9}{'mean':>9}{'sd':>8}{'p10':>8}{'p90':>8}")
    for r in READERS:
        d = np.array([x["end"] - x["start"] for x in by_reader[r]])
        d = d[(d > 0) & (d < 10)]
        say(f"  {r:<8}{len(d):>8}{np.median(d):>9.3f}{d.mean():>9.3f}"
            f"{d.std():>8.3f}{np.percentile(d,10):>8.3f}{np.percentile(d,90):>8.3f}")

    say("\n## 2. CTC alignment confidence (where recorded)")
    for r in READERS:
        s = [x.get("ctc_score") for x in by_reader[r]]
        s = np.array([v for v in s if isinstance(v, (int, float))])
        if len(s):
            say(f"  {r:<8}n={len(s):>7}  median={np.median(s):>8.3f}  "
                f"mean={s.mean():>8.3f}  p10={np.percentile(s,10):>8.3f}")
        else:
            say(f"  {r:<8}(no ctc_score recorded)")

    say("\n## 3. alignment method mix (from the feature meta actually trained on)")
    for r in READERS:
        p = META_DIR / f"{r}_meta.json"
        if not p.exists():
            say(f"  {r:<8}(no meta)")
            continue
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        c = Counter(m.get("align") for m in meta)
        tot = sum(c.values())
        parts = "  ".join(f"{k}={v} ({v/tot*100:.0f}%)" for k, v in c.most_common())
        say(f"  {r:<8}{parts}")

    say("\n## 4. class balance across the top-11 (share of that reader's words)")
    say(f"  {'taam':<16}" + "".join(f"{r:>9}" for r in READERS))
    shares = {}
    for r in READERS:
        c = Counter(x["taam"] for x in by_reader[r] if x["taam"] in TOP11)
        tot = sum(c.values())
        shares[r] = {k: c[k] / tot * 100 for k in TOP11}
    for t in TOP11:
        say(f"  {t:<16}" + "".join(f"{shares[r][t]:>8.1f}%" for r in READERS))

    say("\n## 5. duration separability between classes (one-way F on log duration)")
    say("   higher = this reader's ta'amim differ more in duration, i.e. the")
    say("   acoustic task is intrinsically easier on that reader's own audio")
    say(f"  {'reader':<8}{'F':>10}")
    for r in READERS:
        groups = []
        for t in TOP11:
            d = np.array([x["end"] - x["start"] for x in by_reader[r]
                          if x["taam"] == t])
            d = d[(d > 0.05) & (d < 10)]
            if len(d) > 1:
                groups.append(np.log(d))
        say(f"  {r:<8}{f_statistic(groups):>10.2f}")

    say("\n" + "=" * 72)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
