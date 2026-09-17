"""
Name the invariant: which contour feature, for which ta'am, agrees across readers.
=================================================================================
research_variance.py says HOW MUCH of a ta'am's shape survives crossing
readers. It does not say WHAT survives. This script asks that of thirteen
interpretable scalars (direction, peak position, interval size, final slope,
ornament count, duration, pitch relative to the reader's own tonic), so that
anything found can be written down as a rule a human can check.

THE DECOMPOSITION

Every comparison uses the matched slots -- the same word of the same verse
sung by two different readers -- which is what makes the arithmetic honest:

  bias_max   the largest systematic difference between two readers, estimated
             as the mean of their paired same-word differences. This is the
             dialect term: if it is large, readers genuinely do the ta'am
             differently and no single threshold can serve both.
  noise      SD of those paired differences / sqrt(2) -- one rendition's worth
             of unrepeatable variation, including word-boundary error.
  word_sd    SD across different words within one reader, with the rendition
             noise removed. This is the part driven by the text, not the ta'am.
  d_rest     |mean(this ta'am) - mean(all other ta'amim)| / pooled SD, computed
             SEPARATELY FOR EACH READER, reported as the minimum over readers.
             A feature is only usable as a rule if it works for the WEAKEST
             reader, not on average.

CONFOUND CONTROL

Every feature is also recomputed after regressing out log-duration and
syllable count WITHIN each reader (`_resid` variants). A ta'am whose "rule"
disappears under that was never a melodic rule; it was a statement about word
length. This is the specific failure the project asked to be guarded against.

NULL CONTROL

Labels are shuffled and the whole table recomputed. Everything reported as a
rule must be absent from the shuffled table.

Usage:
  venv/bin/python Data_Prep/research_invariants.py
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
OUT = P / "results" / "melody_research" / "invariants.json"
MIN_PER_READER = 25
MIN_PAIRS = 30
SEED = 0


def book_of(ref):
    return ref.rsplit(" ", 1)[0]


def build(rows):
    """Attach a reader tonic and compute every feature."""
    # tonic = median of the per-word median F0, per (reader, book): a reader's
    # register drifts between recording sessions, and each book is a different
    # set of files
    by_rb = defaultdict(list)
    for r in rows:
        by_rb[(r["reader"], book_of(r["ref"]))].append(r["med_hz"])
    tonic = {k: float(np.median(v)) for k, v in by_rb.items()}

    feats = []
    for r in rows:
        t = tonic[(r["reader"], book_of(r["ref"]))]
        off = 12 * np.log2(r["med_hz"] / t)
        f = features(r["curve"], r["dur"], r.get("stress"), off)
        f["_n_syl"] = (r.get("stress") or {}).get("n_syl", np.nan)
        feats.append(f)
    return feats


def residualise(F, reader, logdur, nsyl):
    """Regress each feature on log-duration and syllable count, per reader."""
    R = F.copy()
    for rd in np.unique(reader):
        m = reader == rd
        ok = m & np.isfinite(nsyl)
        if ok.sum() < 50:
            ok = m
            Xd = np.column_stack([np.ones(ok.sum()), logdur[ok]])
        else:
            Xd = np.column_stack([np.ones(ok.sum()), logdur[ok], nsyl[ok]])
        for c in range(F.shape[1]):
            y = F[ok, c]
            good = np.isfinite(y)
            if good.sum() < 20:
                continue
            beta, *_ = np.linalg.lstsq(Xd[good], y[good], rcond=None)
            R[np.where(ok)[0][good], c] = y[good] - Xd[good] @ beta
    return R


def analyse(F, taam, reader, slot, names, label):
    out = {}
    for t in sorted(set(taam)):
        m = taam == t
        counts = {rd: int((m & (reader == rd)).sum()) for rd in np.unique(reader)}
        readers_ok = [rd for rd, n in counts.items() if n >= MIN_PER_READER]
        if len(readers_ok) < 2:
            continue
        entry = {"n": int(m.sum()), "readers": counts, "features": {}}

        # paired same-slot differences, per reader pair
        slot_idx = defaultdict(dict)
        for i in np.where(m)[0]:
            slot_idx[slot[i]][reader[i]] = i
        pairs = defaultdict(list)
        for s, d in slot_idx.items():
            rs = sorted(d)
            for a in range(len(rs)):
                for b in range(a + 1, len(rs)):
                    if rs[a] in readers_ok and rs[b] in readers_ok:
                        pairs[(rs[a], rs[b])].append((d[rs[a]], d[rs[b]]))
        pairs = {k: v for k, v in pairs.items() if len(v) >= MIN_PAIRS}
        if not pairs:
            continue

        for ci, name in enumerate(names):
            col = F[:, ci]
            rest = col[~m & np.isfinite(col)]
            if len(rest) < 50:
                continue
            bias, noise_v, npair = {}, [], 0
            for (ra, rb), lst in pairs.items():
                d = np.array([col[i] - col[j] for i, j in lst])
                d = d[np.isfinite(d)]
                if len(d) < MIN_PAIRS:
                    continue
                bias[f"{ra}|{rb}"] = float(d.mean())
                noise_v.append(d.var())
                npair += len(d)
            if not bias:
                continue
            noise = float(np.sqrt(np.mean(noise_v) / 2))

            per_reader = {}
            for rd in readers_ok:
                v = col[m & (reader == rd)]
                v = v[np.isfinite(v)]
                w = col[(~m) & (reader == rd)]
                w = w[np.isfinite(w)]
                if len(v) < MIN_PER_READER or len(w) < 30:
                    continue
                sp = np.sqrt((v.var() + w.var()) / 2)
                per_reader[rd] = {
                    "mean": float(v.mean()), "sd": float(v.std()),
                    "d_rest": float((v.mean() - w.mean()) / sp) if sp > 1e-9 else 0.0,
                    "n": int(len(v)),
                }
            if len(per_reader) < 2:
                continue
            drs = [abs(x["d_rest"]) for x in per_reader.values()]
            sds = [x["sd"] for x in per_reader.values()]
            word_var = float(np.mean([s ** 2 for s in sds])) - noise ** 2
            entry["features"][name] = {
                "bias_max": float(max(abs(v) for v in bias.values())),
                "bias": bias, "noise": noise,
                "word_sd": float(np.sqrt(max(word_var, 0.0))),
                "d_rest_min": float(min(drs)), "d_rest_mean": float(np.mean(drs)),
                "sign_agree": bool(len({np.sign(x["d_rest"]) for x in per_reader.values()}) == 1),
                "per_reader": per_reader, "n_pairs": npair,
                "rest_sd": float(rest.std()),
            }
        out[t] = entry
    return out


def main():
    rows = json.loads(CURVES.read_text(encoding="utf-8"))
    print(f"{len(rows):,} curves")
    feats = build(rows)
    names = FEATURE_NAMES + ["stress_rel", "peak_minus_stress"]
    F = np.array([[f.get(n, np.nan) for n in names] for f in feats], float)
    taam = np.array([r["taam"] for r in rows])
    reader = np.array([r["reader"] for r in rows])
    slot = np.array([r["slot_id"] for r in rows])
    logdur = np.log(np.array([r["dur"] for r in rows]))
    nsyl = np.array([f["_n_syl"] for f in feats], float)

    real = analyse(F, taam, reader, slot, names, "real")
    Fr = residualise(F, reader, logdur, nsyl)
    resid = analyse(Fr, taam, reader, slot, names, "resid")

    # Null control. The label must be permuted AT THE SLOT LEVEL: every reader
    # of one slot keeps the same (fake) ta'am, so the paired same-word
    # structure the whole analysis rests on is preserved and only the
    # ta'am-to-word assignment is destroyed. Permuting per-curve instead
    # silently empties the pair sets and reports NaN, which is not a control.
    rng = np.random.default_rng(SEED)
    slots_u = np.unique(slot)
    slot_taam = {s: taam[np.where(slot == s)[0][0]] for s in slots_u}
    fake_vals = rng.permutation([slot_taam[s] for s in slots_u])
    fake = {s: v for s, v in zip(slots_u, fake_vals)}
    shuf = analyse(F, np.array([fake[s] for s in slot]), reader, slot, names, "shuf")

    OUT.write_text(json.dumps({"real": real, "resid": resid, "shuffled": shuf,
                               "names": names}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"-> {OUT.relative_to(P)}")

    # quick console view: strongest usable rule per ta'am
    print(f"\n{'taam':<18}{'feature':<18}{'d_rest_min':>11}{'bias_max':>10}"
          f"{'noise':>8}{'word_sd':>9}{'resid_d':>9}{'shuf_d':>8}")
    for t, e in sorted(real.items()):
        best, bestv = None, 0
        for name, f in e["features"].items():
            if name in ("dur", "stress_rel", "_n_syl"):
                continue
            if not f["sign_agree"]:
                continue
            if f["d_rest_min"] > bestv:
                best, bestv = name, f["d_rest_min"]
        if best is None:
            print(f"{t:<18}{'-- none --':<18}")
            continue
        f = e["features"][best]
        rd = resid.get(t, {}).get("features", {}).get(best, {}).get("d_rest_min", float('nan'))
        sh = shuf.get(t, {}).get("features", {}).get(best, {}).get("d_rest_min", float('nan'))
        print(f"{t:<18}{best:<18}{f['d_rest_min']:>11.2f}{f['bias_max']:>10.2f}"
              f"{f['noise']:>8.2f}{f['word_sd']:>9.2f}{rd:>9.2f}{sh:>8.2f}")


if __name__ == "__main__":
    main()
