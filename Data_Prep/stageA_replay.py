"""
Stage A replay (H0), stratified across ALL eight held-out parashot.
===================================================================

WHY THIS EXISTS

`real_block_clips(limit)` takes the first `limit` eligible blocks in
metadata order -- no sampling. Held-out blocks are ordered by parasha, so:

  real_block_clips(300)  ->  parasha 5 only
  real_block_clips(900)  ->  parashot 5, 6, 8 only (all Genesis)

A1/A2/A5 in PLAN_STAGE_A.md were measured on the 300 and the served
thresholds (A3/A4) derived on the 900. Nothing leaked -- these are held-out
parashot -- but "measured on the eight held-out parashot" was one parasha.
Rerunning the old job reproduces its numbers to the last digit because it
replays the same blocks through a deterministic model; that confirms nothing
broke, not that the number holds across the holdout.

This draws an equal number of blocks per held-out parasha with a recorded
seed, posts them to the live API exactly as r1_stageA_reverify did, and
scores against the SERVED thresholds (not percentiles re-derived on the
same sample), reporting per parasha so a one-parasha number cannot hide.

Negatives are the same proxy as before: the true audio sent with each
expected taam swapped for its most-confused partner. That is a reader who
performed X when Y was expected -- a substitution, not an authentic learner
error. PRODUCT_ACCEPTANCE.md requires authentic errors for the recall gate,
so A4 here remains a proxy.

Needs the server on :8000.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_end_to_end as C   # noqa: E402

P = Path(__file__).resolve().parent.parent


def stratified_blocks(per_parasha, seed, min_words=2):
    """Same exact verse+block join as real_block_clips, grouped by parasha."""
    meta = json.loads((P / "features_v3_word" / "BEN13R_meta.json")
                      .read_text(encoding="utf-8"))
    man = json.loads((P / "data" / "ben13_reading" /
                      "manifest_yerushalmi.json").read_text(encoding="utf-8"))
    by_block = {(r["book"], r["perek"], r["pasuk"], str(r["block"])): r["local_path"]
                for r in man}
    blocks = defaultdict(list)
    for i, m in enumerate(meta):
        blocks[(m["verse_ref"], m["segment_block"])].append(i)
    held = set(C.HELD_OUT)
    pool = defaultdict(list)
    for (ref, blk), idxs in blocks.items():
        par = str(meta[idxs[0]].get("parasha"))
        if len(idxs) < min_words or par not in held:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["start"])
        b, c, v = ref.split(".")
        path = by_block.get((b, c, v, str(blk)))
        if not path or not (P / path).exists():
            continue
        pool[par].append((str(P / path), [meta[i]["word_raw"] for i in idxs],
                          [meta[i]["taam"] for i in idxs]))
    rng = random.Random(seed)
    out = []
    for par in sorted(pool, key=int):
        for clip in rng.sample(pool[par], min(per_parasha, len(pool[par]))):
            out.append((par,) + clip)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-parasha", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default="results/research_os/audit/stageA_replay.json")
    args = ap.parse_args()

    C.CONFUSION = C._load_confusion()
    cal = json.loads((P / "results" / "calibration" / "calibration_e2e_fixed.json")
                     .read_text(encoding="utf-8"))
    wrong_below = cal["say_wrong_below"]

    clips = stratified_blocks(args.per_parasha, args.seed)
    per = defaultdict(lambda: {"blocks": 0, "gated": 0, "words": 0, "correct": 0,
                               "pos": [], "neg": []})
    for par, path, words, taamim in clips:
        s = per[par]
        s["blocks"] += 1
        d = C.post(path, words, taamim)
        if not d or not d.get("aligned"):
            s["gated"] += 1
            continue
        swapped = []
        for w in d.get("words", []):
            if w.get("confidence") is None:
                continue
            s["pos"].append(w["confidence"])
            s["words"] += 1
            s["correct"] += w.get("heard") == w.get("expected")
            swapped.append(C.CONFUSION.get(w.get("expected")))
        if all(swapped) and len(swapped) == len(taamim):
            dn = C.post(path, words, swapped)
            for w in (dn or {}).get("words", []):
                if w.get("confidence") is not None:
                    s["neg"].append(w["confidence"])

    def summarise(s):
        pos, neg = np.array(s["pos"]), np.array(s["neg"])
        n = min(len(pos), len(neg))
        return {
            "blocks": s["blocks"], "gated": s["gated"], "words": s["words"],
            "A1_accuracy": s["correct"] / max(s["words"], 1),
            "A2_auc": (float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)],
                                           np.r_[pos[:n], neg[:n]])) if n else None),
            "A3_contradict_correct": float((pos < wrong_below).mean()) if len(pos) else None,
            "A4_catch_proxy": float((neg < wrong_below).mean()) if len(neg) else None,
            "A5_zeros": float((pos < 0.01).mean()) if len(pos) else None,
            "n_neg": len(neg),
        }

    total = {"blocks": 0, "gated": 0, "words": 0, "correct": 0, "pos": [], "neg": []}
    for s in per.values():
        for k in total:
            total[k] += s[k]
    result = {
        "seed": args.seed, "per_parasha_target": args.per_parasha,
        "served_say_wrong_below": wrong_below,
        "negatives": "confusion-partner swap of expected taam (proxy, not authentic errors)",
        "overall": summarise(total),
        "by_parasha": {par: summarise(per[par]) for par in sorted(per, key=int)},
    }
    out = P / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def row(name, r):
        f = lambda v: "  -  " if v is None else f"{v*100:5.1f}"
        auc = "  -  " if r["A2_auc"] is None else f"{r['A2_auc']:.3f}"
        print(f"  {name:>8} {r['blocks']:4d} {r['gated']:4d} {r['words']:5d}  "
              f"{f(r['A1_accuracy'])}  {auc}  {f(r['A3_contradict_correct'])}  "
              f"{f(r['A4_catch_proxy'])}  {f(r['A5_zeros'])}")
    print(f"  seed={args.seed}  served say_wrong_below={wrong_below}")
    print("  parasha  blk gate words     A1    A2      A3     A4     A5")
    for par, r in result["by_parasha"].items():
        row(par, r)
    row("ALL", result["overall"])


if __name__ == "__main__":
    main()
