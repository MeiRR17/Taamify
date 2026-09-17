"""
Same word, same ta'am vs same word, different ta'am -- within and across readers.
================================================================================
Fixing the word fixes the syllable count, the stress position and the phonetic
material, so the only thing left varying is the ta'am (and, in the cross-reader
condition, the singer). The question becomes a discrimination:

    given two renditions of the SAME WORD, can the contour distance tell
    whether they carry the same ta'am?

  AUC_within   both renditions by one reader, two different verses.
               The single-voice ceiling: what a perfect Ben13-only model
               could ever achieve.
  AUC_cross    the two renditions by two different readers.
               What a product that judges a stranger against Ben13 faces.

  portability  (AUC_cross - 0.5) / (AUC_within - 0.5)

0.5 is chance, and the shuffled control (ta'am labels permuted within each
word, so the same pairs are re-used with meaningless labels) must land there.

Duration is reported alongside, because a disjunctive is longer than a
conjunctive on the same word and a distance metric can read length instead of
melody. The `_durmatched` columns repeat everything on the subset of pairs
whose durations are within 15% of each other.

Usage:
  venv/bin/python Data_Prep/research_wordtest.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from research_melody_lib import dtw_pairs  # noqa: E402

MR = P / "results" / "melody_research"
CURVES = MR / "curves_wordpairs.json"
OUT = MR / "wordtest.json"
MAX_PAIRS = 4000
SEED = 0


def auc_pairs(same, diff):
    """P(d_same < d_diff), ties at 0.5."""
    if len(same) < 25 or len(diff) < 25:
        return float("nan"), 0
    from scipy.stats import rankdata
    a = np.concatenate([same, diff])
    r = rankdata(a, method="average")
    n1, n0 = len(same), len(diff)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(1 - u / (n1 * n0)), min(n1, n0)


def collect(rows, labels, rng):
    """For each ta'am: four distance populations, at fixed word."""
    by_word = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_word[r["word_clean"]][labels[i]].append(i)

    pops = defaultdict(lambda: defaultdict(list))
    reader = [r["reader"] for r in rows]
    for w, cells in by_word.items():
        ts = [t for t, v in cells.items() if len(v) >= 2]
        if len(ts) < 2:
            continue
        for t in ts:
            ids = cells[t]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i, j = ids[a], ids[b]
                    key = "same_within" if reader[i] == reader[j] else "same_cross"
                    pops[t][key].append((i, j))
            for t2 in ts:
                if t2 == t:
                    continue
                for i in ids:
                    for j in cells[t2]:
                        key = "diff_within" if reader[i] == reader[j] else "diff_cross"
                        pops[t][key].append((i, j))
    return pops


def main():
    rows = json.loads(CURVES.read_text(encoding="utf-8"))
    print(f"{len(rows):,} curves, {len({r['word_clean'] for r in rows}):,} words")
    C = np.ascontiguousarray(np.array([r["curve"] for r in rows], float))
    dur = np.array([r["dur"] for r in rows])
    taam = np.array([r["taam"] for r in rows])
    rng = np.random.default_rng(SEED)

    def evaluate(labels, tag):
        pops = collect(rows, labels, rng)
        table = {}
        for t, d in sorted(pops.items()):
            cnt = {k: len(v) for k, v in d.items()}
            if min(cnt.get(k, 0) for k in ("same_within", "diff_within",
                                           "same_cross", "diff_cross")) < 25:
                continue
            dist = {}
            for k, lst in d.items():
                sel = lst if len(lst) <= MAX_PAIRS else [
                    lst[i] for i in rng.choice(len(lst), MAX_PAIRS, replace=False)]
                ia = np.array([s[0] for s in sel])
                ib = np.array([s[1] for s in sel])
                dd = dtw_pairs(C, C, ia, ib)
                keep = np.abs(dur[ia] - dur[ib]) <= 0.15 * np.minimum(dur[ia], dur[ib])
                dist[k] = (dd, dd[keep])
            aw, nw = auc_pairs(dist["same_within"][0], dist["diff_within"][0])
            ac, nc = auc_pairs(dist["same_cross"][0], dist["diff_cross"][0])
            awd, _ = auc_pairs(dist["same_within"][1], dist["diff_within"][1])
            acd, _ = auc_pairs(dist["same_cross"][1], dist["diff_cross"][1])
            port = ((ac - 0.5) / (aw - 0.5)) if (aw - 0.5) > 0.02 else float("nan")
            table[t] = {"auc_within": aw, "auc_cross": ac,
                        "auc_within_durmatched": awd, "auc_cross_durmatched": acd,
                        "portability": port, "n_pairs": cnt}
        print(f"\n== {tag}")
        print(f"  {'taam':<18}{'within':>8}{'cross':>8}{'port':>8}"
              f"{'within_dm':>11}{'cross_dm':>10}{'pairs':>9}")
        for t, e in table.items():
            print(f"  {t:<18}{e['auc_within']:>8.3f}{e['auc_cross']:>8.3f}"
                  f"{e['portability']:>8.2f}{e['auc_within_durmatched']:>11.3f}"
                  f"{e['auc_cross_durmatched']:>10.3f}"
                  f"{min(e['n_pairs'].values()):>9}")
        if table:
            print(f"  MEAN within {np.nanmean([e['auc_within'] for e in table.values()]):.3f}"
                  f"  cross {np.nanmean([e['auc_cross'] for e in table.values()]):.3f}"
                  f"  within_dm {np.nanmean([e['auc_within_durmatched'] for e in table.values()]):.3f}"
                  f"  cross_dm {np.nanmean([e['auc_cross_durmatched'] for e in table.values()]):.3f}")
        return table

    real = evaluate(taam, "real")
    # null: permute the ta'am label within each word, so the same pairs recur
    # with meaningless labels
    lab = taam.copy()
    by_word = defaultdict(list)
    for i, r in enumerate(rows):
        by_word[r["word_clean"]].append(i)
    for w, ids in by_word.items():
        lab[ids] = rng.permutation(taam[ids])
    shuf = evaluate(lab, "shuffled-within-word control")

    OUT.write_text(json.dumps({"real": real, "shuffled": shuf},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(P)}")


if __name__ == "__main__":
    main()
