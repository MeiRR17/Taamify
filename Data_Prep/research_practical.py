"""
Does a per-ta'am "core" score transfer to new readers better than whole-contour DTW?
===================================================================================
The product's failure mode, stated exactly: templates come from Ben13, the user
is not Ben13. So the test is built that way -- every reference is Ben13, every
test item is R1/R2/R8, and the same experiment is repeated with held-out BEN13
test items so the transfer drop can be read off directly.

Negatives are built the way build_stage_b.py builds them: a real recording of a
real word sung with ta'am A, presented where the text requires ta'am B, with B
drawn from the ta'amim that this same word actually carries elsewhere in the
Torah. Nothing is synthesised.

Four scores, same positives and negatives:
  dtw_medoid  -DTW to the Ben13 medoid of the required ta'am (what
              webapp/server.py's get_class_medoids does today)
  dtw_knn5    -mean of the 5 smallest DTW distances to Ben13 exemplars
  core_z      -mean |z| over the features that research_rules.py found to be
              cross-reader invariant for that ta'am, z-scored on BEN13 only
  core_rule   fraction of that ta'am's passing rules the item satisfies

Reported as AUC of separating correct from wrong, per ta'am. AUC (not accuracy)
because the two scores live on different scales and no threshold is being
claimed.

Usage:
  venv/bin/python Data_Prep/research_practical.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from research_melody_lib import features, FEATURE_NAMES, dtw_pairs  # noqa: E402

MR = P / "results" / "melody_research"
OUT = MR / "practical.json"
OTHERS = ["R1", "R2", "R8"]
K = 5
N_REF = 80            # Ben13 exemplars per class
SEED = 0


WINDOW_SCALED = {"turns", "range", "sd", "dur", "rise_first"}


def tier(rules, t):
    """Tier A rules (research_rules.py's bar) if any; else the Tier B ones, so
    every ta'am gets the strongest core score its own data supports and the
    comparison against DTW is not decided by an empty feature list."""
    a = rules["verdict"].get(t, {}).get("rules", [])
    if a:
        return a
    out = []
    for name, r in rules["real"].get(t, {}).items():
        if name in WINDOW_SCALED or r["lift"] is None:
            continue
        sl = rules["shuffled"].get(t, {}).get(name, {}).get("lift")
        if sl is not None and sl > 1.1:
            continue
        if r["hit_min"] >= 0.60 and r["lift"] >= 1.2 and r["spread"] <= 0.30:
            out.append({"feature": name, **r})
    out.sort(key=lambda r: -r["lift"])
    return out[:3]


def auc(pos, neg):
    if len(pos) < 10 or len(neg) < 10:
        return float("nan")
    # mid-ranks, not argsort-of-argsort: core_rule only takes the values
    # {0, 1/3, 2/3, 1}, so ties are the norm for it, and a stable argsort would
    # silently break every tie in favour of whichever array was concatenated
    # first -- inflating exactly the score being argued for.
    from scipy.stats import rankdata
    a = np.concatenate([pos, neg])
    r = rankdata(a, method="average")
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    rows = json.loads((MR / "curves_matched.json").read_text(encoding="utf-8"))
    rules = json.loads((MR / "rules.json").read_text(encoding="utf-8"))
    index = json.loads((MR / "matched_index.json").read_text(encoding="utf-8"))

    # which ta'amim does each word actually carry -> honest wrong-ta'am pool
    word_taams = defaultdict(set)
    for s in index:
        word_taams[s["word_clean"]].add(s["taam"])

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
    F = np.array(F, float)
    C = np.ascontiguousarray(np.array([r["curve"] for r in rows], float))
    taam = np.array([r["taam"] for r in rows])
    reader = np.array([r["reader"] for r in rows])
    rng = np.random.default_rng(SEED)

    # ---- Ben13 reference bank, and a held-out Ben13 test split
    b13 = np.where(reader == "BEN13R")[0]
    rng.shuffle(b13)
    ref_idx, b13_test = defaultdict(list), []
    for i in b13:
        if len(ref_idx[taam[i]]) < N_REF:
            ref_idx[taam[i]].append(int(i))
        else:
            b13_test.append(int(i))
    ref_idx = {t: np.array(v) for t, v in ref_idx.items() if len(v) >= 20}

    # medoid of each class among the reference bank
    medoid = {}
    for t, ids in ref_idx.items():
        n = len(ids)
        ia = np.repeat(np.arange(n), n)
        ib = np.tile(np.arange(n), n)
        d = dtw_pairs(C[ids], C[ids], ia, ib).reshape(n, n)
        medoid[t] = int(ids[int(np.argmin(d.sum(1)))])

    # Ben13 feature stats and the invariant feature list per ta'am
    stats, keep = {}, {}
    for t, ids in ref_idx.items():
        mu = np.nanmean(F[ids], 0)
        sd = np.nanstd(F[ids], 0) + 1e-6
        stats[t] = (mu, sd)
        keep[t] = [names.index(r["feature"]) for r in tier(rules, t)
                   if r["feature"] in names]

    def score_item(i, t):
        out = {}
        out["dtw_medoid"] = -float(dtw_pairs(C, C, np.array([i]),
                                             np.array([medoid[t]]))[0])
        ids = ref_idx[t]
        d = dtw_pairs(C, C, np.full(len(ids), i), ids)
        out["dtw_knn5"] = -float(np.sort(d)[:K].mean())
        mu, sd = stats[t]
        cols = keep[t]
        if cols:
            z = np.abs((F[i, cols] - mu[cols]) / sd[cols])
            out["core_z"] = -float(np.nanmean(z))
            v = tier(rules, t)
            hit = []
            for r in v:
                ci = names.index(r["feature"])
                x = F[i, ci]
                if not np.isfinite(x):
                    continue
                hit.append(x >= r["threshold"] if r["direction"] == ">="
                           else x <= r["threshold"])
            out["core_rule"] = float(np.mean(hit)) if hit else float("nan")
        else:
            out["core_z"] = float("nan")
            out["core_rule"] = float("nan")
        return out

    def run(test_idx, label):
        res = defaultdict(lambda: {"pos": defaultdict(list), "neg": defaultdict(list)})
        for i in test_idx:
            t = taam[i]
            if t not in ref_idx:
                continue
            for k, v in score_item(int(i), t).items():
                res[t]["pos"][k].append(v)
            alt = [x for x in word_taams.get(rows[i]["word_clean"], set())
                   if x != t and x in ref_idx]
            if not alt:
                continue
            tw = alt[rng.integers(len(alt))]
            for k, v in score_item(int(i), tw).items():
                res[tw]["neg"][k].append(v)
        table = {}
        for t, d in res.items():
            e = {"n_pos": len(d["pos"]["dtw_medoid"]),
                 "n_neg": len(d["neg"]["dtw_medoid"])}
            for k in ("dtw_medoid", "dtw_knn5", "core_z", "core_rule"):
                pos = np.array([x for x in d["pos"][k] if np.isfinite(x)])
                neg = np.array([x for x in d["neg"][k] if np.isfinite(x)])
                e[k] = auc(pos, neg)
            table[t] = e
        print(f"\n== {label}")
        print(f"  {'taam':<18}{'n+':>6}{'n-':>6}{'medoid':>9}{'knn5':>9}"
              f"{'core_z':>9}{'core_rule':>11}")
        for t, e in sorted(table.items()):
            if e["n_pos"] < 25 or e["n_neg"] < 25:
                continue
            print(f"  {t:<18}{e['n_pos']:>6}{e['n_neg']:>6}{e['dtw_medoid']:>9.3f}"
                  f"{e['dtw_knn5']:>9.3f}{e['core_z']:>9.3f}{e['core_rule']:>11.3f}")
        ok = [e for e in table.values() if e["n_pos"] >= 25 and e["n_neg"] >= 25]
        for k in ("dtw_medoid", "dtw_knn5", "core_z", "core_rule"):
            vals = [e[k] for e in ok if np.isfinite(e[k])]
            print(f"  mean {k:<12}{np.mean(vals):.3f}  ({len(vals)} ta'amim)")
        return table

    other_idx = np.where(np.isin(reader, OTHERS))[0]
    rng.shuffle(other_idx)
    t_other = run(other_idx[:6000], "test on R1/R2/R8 (references are Ben13)")
    t_b13 = run(np.array(b13_test[:3000]), "control: test on held-out BEN13")

    # ---------------------------------------------------------------- LORO
    # The numbers above have one leak: research_rules.py decided WHICH rules
    # count by checking them on R1/R2/R8, and the same R1/R2/R8 items are then
    # scored. Redo it leave-one-reader-out -- thresholds still fitted on Ben13
    # alone, the pass/fail decision made on Ben13 plus two readers, and the
    # third reader never seen until it is scored.
    print("\n" + "=" * 72)
    print("leave-one-reader-out rule selection (no leak)")
    loro = {}
    for held in OTHERS:
        seen = [r for r in OTHERS if r != held]
        sel = {}
        for t in ref_idx:
            cand = []
            for name, r in rules["real"].get(t, {}).items():
                if name in WINDOW_SCALED or r["lift"] is None:
                    continue
                sl = rules["shuffled"].get(t, {}).get(name, {}).get("lift")
                if sl is not None and sl > 1.1:
                    continue
                hits = [r["hit"][x] for x in seen]
                if (min(hits) >= 0.60
                        and max(hits + [r["hit"]["BEN13R"]]) -
                        min(hits + [r["hit"]["BEN13R"]]) <= 0.30
                        and r["lift"] >= 1.2):
                    cand.append({"feature": name, **r})
            cand.sort(key=lambda r: -r["lift"])
            sel[t] = cand[:3]
        keep.clear()
        keep.update({t: [names.index(r["feature"]) for r in sel[t]]
                     for t in ref_idx})
        rules_backup = rules["verdict"]
        rules["verdict"] = {t: {"rules": sel[t]} for t in sel}
        idx = np.where(reader == held)[0]
        rng.shuffle(idx)
        loro[held] = run(idx[:2500], f"held-out reader {held}")
        rules["verdict"] = rules_backup

    print("\nLORO summary (ta'amim where a rule set existed for that fold):")
    for k in ("dtw_medoid", "dtw_knn5", "core_z", "core_rule"):
        vals = []
        for held, tab in loro.items():
            for t, e in tab.items():
                if e["n_pos"] >= 25 and e["n_neg"] >= 25 and np.isfinite(e["core_rule"]):
                    vals.append(e[k])
        print(f"  {k:<12}{np.mean(vals):.3f}  (n={len(vals)} ta'am-folds)")

    OUT.write_text(json.dumps({"others": t_other, "ben13_heldout": t_b13,
                               "loro": loro}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n-> {OUT.relative_to(P)}")


if __name__ == "__main__":
    main()
