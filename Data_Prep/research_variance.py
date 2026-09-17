"""
Decompose ta'am contour variation into within-reader and between-reader parts.
=============================================================================
THE DESIGN

Three DTW distance populations per ta'am, all on the same curve set:

  D_self   two renditions by the SAME reader of DIFFERENT words carrying this
           ta'am. How much one person varies across instances -- the ceiling
           on how tight any template can be even for a single voice.
  D_cross  two DIFFERENT readers' renditions of the SAME WORD of the SAME
           VERSE. Word length, stress position and position-in-verse are held
           fixed by construction, so whatever is left is the reader.
  D_null   two renditions of DIFFERENT ta'amim, DURATION-MATCHED (within 15%)
           and reader-matched. This is the floor: how far apart two unrelated
           chunks of this reader's chanting of this length happen to be.

Two normalised readings follow:

  identity    I = (D_null - D_self) / D_null
              how much of the unrelated-pair distance a ta'am's own shape
              recovers within one voice. I ~ 0 means the ta'am has no melodic
              identity at all, not even for one reader.
  portability P = (D_null - D_cross) / (D_null - D_self)
              the fraction of that identity that survives crossing readers.
              P ~ 1 means readers realise the ta'am the same way; P ~ 0 means
              each reader has his own version of it.

CONTROLS (this project has twice been fooled by a missing one)

  shuffle  ta'am labels permuted across curves, everything recomputed. If the
           pipeline manufactures structure, I and P stay high here.
  dur-null the null is duration-matched, so "this ta'am lands on long words"
           cannot masquerade as a melodic invariant.
  b13-only D_self is also computed for Ben13 alone, whose word boundaries come
           from segment-level CTC rather than verse-level, to check that the
           cross-reader numbers are not just R1/R2/R8 boundary noise.

Usage:
  venv/bin/python Data_Prep/research_variance.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from research_melody_lib import dtw_pairs, READERS  # noqa: E402

CURVES = P / "results" / "melody_research" / "curves_matched.json"
OUT = P / "results" / "melody_research" / "variance.json"
N_PAIRS = 3000          # per population per ta'am
MIN_N = 40
SEED = 0


def sample_pairs(rng, groups, n):
    """(i, j) pairs drawn from within each group, pooled."""
    keys = [k for k, v in groups.items() if len(v) >= 2]
    if not keys:
        return np.zeros(0, int), np.zeros(0, int)
    w = np.array([len(groups[k]) for k in keys], float)
    w /= w.sum()
    ia, ib = [], []
    for _ in range(n):
        k = keys[rng.choice(len(keys), p=w)]
        g = groups[k]
        a, b = rng.choice(len(g), 2, replace=False)
        ia.append(g[a])
        ib.append(g[b])
    return np.array(ia), np.array(ib)


def run(rows, C, taams, label, rng):
    dur = np.array([r["dur"] for r in rows])
    reader = np.array([r["reader"] for r in rows])
    out = {}
    for t in taams:
        idx_t = np.where(taams_arr(rows) == t)[0]
        if len(idx_t) < MIN_N:
            continue
        res = {"n_curves": int(len(idx_t))}

        # ---- D_self: same reader, same ta'am, different word
        g = defaultdict(list)
        for i in idx_t:
            g[reader[i]].append(int(i))
        ia, ib = sample_pairs(rng, g, N_PAIRS)
        if len(ia) == 0:
            continue
        d_self = dtw_pairs(C, C, ia, ib)
        res["d_self"] = float(d_self.mean())
        res["d_self_n"] = int(len(ia))
        # Ben13 alone (cleanest boundaries)
        gb = {k: v for k, v in g.items() if k == "BEN13R" and len(v) >= 2}
        if gb:
            ia2, ib2 = sample_pairs(rng, gb, min(N_PAIRS, 1500))
            res["d_self_ben13"] = float(dtw_pairs(C, C, ia2, ib2).mean())

        # ---- D_cross: same slot, different readers
        gs = defaultdict(list)
        for i in idx_t:
            gs[rows[i]["slot_id"]].append(int(i))
        gs = {k: v for k, v in gs.items() if len(v) >= 2}
        ia, ib = sample_pairs(rng, gs, N_PAIRS)
        if len(ia) == 0:
            continue
        d_cross = dtw_pairs(C, C, ia, ib)
        res["d_cross"] = float(d_cross.mean())
        res["d_cross_n"] = int(len(ia))
        res["n_slots"] = int(len(gs))

        # ---- D_null: different ta'am, same reader, duration-matched
        others = np.where(taams_arr(rows) != t)[0]
        ia, ib = [], []
        by_reader_other = defaultdict(list)
        for i in others:
            by_reader_other[reader[i]].append(int(i))
        tries = 0
        while len(ia) < N_PAIRS and tries < N_PAIRS * 40:
            tries += 1
            i = int(rng.choice(idx_t))
            pool = by_reader_other.get(reader[i])
            if not pool:
                continue
            j = int(pool[rng.integers(len(pool))])
            if abs(dur[j] - dur[i]) <= 0.15 * dur[i]:
                ia.append(i)
                ib.append(j)
        if len(ia) < 200:
            continue
        d_null = dtw_pairs(C, C, np.array(ia), np.array(ib))
        res["d_null"] = float(d_null.mean())
        res["d_null_n"] = int(len(ia))

        res["identity"] = (res["d_null"] - res["d_self"]) / res["d_null"]
        denom = res["d_null"] - res["d_self"]
        res["portability"] = ((res["d_null"] - res["d_cross"]) / denom
                              if denom > 1e-9 else float("nan"))

        # per-reader-pair cross distance -- is one reader the outlier?
        pairdist = defaultdict(list)
        for k, v in gs.items():
            for a in range(len(v)):
                for b in range(a + 1, len(v)):
                    ra, rb = reader[v[a]], reader[v[b]]
                    pairdist[tuple(sorted((ra, rb)))].append((v[a], v[b]))
        pp = {}
        for k, lst in pairdist.items():
            if len(lst) < 30:
                continue
            sel = [lst[i] for i in rng.choice(len(lst), min(1200, len(lst)),
                                              replace=False)]
            aa = np.array([s[0] for s in sel])
            bb = np.array([s[1] for s in sel])
            pp["|".join(k)] = [float(dtw_pairs(C, C, aa, bb).mean()), len(lst)]
        res["reader_pairs"] = pp
        out[t] = res
        print(f"  [{label}] {t:<18} n={res['n_curves']:>5} slots={res['n_slots']:>4} "
              f"self={res['d_self']:.3f} cross={res['d_cross']:.3f} "
              f"null={res['d_null']:.3f}  I={res['identity']:.3f} "
              f"P={res['portability']:.3f}", flush=True)
    return out


_TA = {}


def taams_arr(rows):
    key = id(rows)
    if key not in _TA:
        _TA[key] = np.array([r["taam"] for r in rows])
    return _TA[key]


def main():
    rows = json.loads(CURVES.read_text(encoding="utf-8"))
    print(f"{len(rows):,} curves loaded")
    C = np.ascontiguousarray(np.array([r["curve"] for r in rows], dtype=np.float64))

    # reader tonic: the median of each reader's per-word median F0
    from collections import Counter
    print("  " + str(Counter(r["reader"] for r in rows).most_common()))

    taams = sorted({r["taam"] for r in rows})
    rng = np.random.default_rng(SEED)
    real = run(rows, C, taams, "real", rng)

    # ---- shuffle control: permute ta'am labels, keep everything else
    print("\nshuffled-label control")
    perm = rng.permutation(len(rows))
    srows = [dict(r, taam=rows[perm[i]]["taam"]) for i, r in enumerate(rows)]
    shuf = run(srows, C, taams, "shuf", np.random.default_rng(SEED + 1))

    OUT.write_text(json.dumps({"real": real, "shuffled": shuf},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(P)}")

    iv = [v["identity"] for v in real.values()]
    isv = [v["identity"] for v in shuf.values()]
    print(f"\nidentity   real mean {np.mean(iv):.3f}   shuffled mean {np.mean(isv):.3f}")
    pv = [v["portability"] for v in real.values()]
    psv = [v["portability"] for v in shuf.values()]
    print(f"portability real mean {np.nanmean(pv):.3f}   shuffled mean {np.nanmean(psv):.3f}")


if __name__ == "__main__":
    main()
