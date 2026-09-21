"""
Derive a genuine two-threshold policy: wrong / neutral / correct.
==================================================================
The calibration currently in production collapsed to a single number
(`say_wrong_below == say_correct_above == 0.3497`), which means the neutral
state does not exist: every word is forced into either "correct" or "wrong".
The product spec is built on three states, so this derives the missing band.

The two thresholds answer two different questions, and they are set against
two different error budgets:

  say_wrong_below   — below this we are willing to tell a learner they were
                      wrong. Chosen so that at most CONTRADICT_BUDGET of
                      genuinely CORRECT readings fall below it. This is the
                      expensive error: a child contradicted while right stops
                      believing the feedback.

  say_correct_above — above this we are willing to confirm. Chosen so that at
                      most BLESS_BUDGET of real, confusable MISTAKES land
                      above it, because confirming an actual error teaches
                      the mistake.

Anything between is neutral, and the honest thing to show is a grey mark and
"I did not hear that clearly" — not a verdict.

Negatives are the CONFUSABLE class (the one the CNN itself most often
mistakes the true one for), not a random wrong ta'am. A learner's error is
never random, and calibrating against random negatives produces a threshold
that looks excellent and fails in the field.

Usage:
  python Data_Prep/derive_neutral_band.py
  python Data_Prep/derive_neutral_band.py --head results/binary_verify/verify_head_hard0.75.pth
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
FEAT = P / "features_v3_word"
OUT = P / "results" / "calibration"

CONTRADICT_BUDGET = 0.02   # at most 2% of correct readings may be called wrong
BLESS_BUDGET = 0.10        # at most 10% of real mistakes may be confirmed
N_WORDS = 2500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("srv", P / "webapp" / "server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    if args.head:
        srv.VERIFY_HEAD_PATH = Path(args.head)
    st = srv.get_model()
    classes = list(st["class_names"])

    X = np.load(FEAT / "BEN13R_X.npy", mmap_mode="r")
    meta = json.loads((FEAT / "BEN13R_meta.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    sys.path.insert(0, str(P / "Data_Prep"))
    from holdout import HELD_OUT
    held = set(HELD_OUT)
    idx = [i for i, m in enumerate(meta)
           if str(m.get("parasha")) in held and m["taam"] in classes]
    rng.shuffle(idx)
    idx = idx[:N_WORDS]
    print(f"{len(idx):,} words from {len(held)} held-out parashot")

    conf = defaultdict(Counter)
    pos = []
    for n, i in enumerate(idx, 1):
        s = np.array(X[i])
        true = meta[i]["taam"]
        top3, _ = srv.classify(s)
        if top3 and top3[0][0] != true:
            conf[true][top3[0][0]] += 1
        pos.append(srv.verify_match(s, true)[1])
        if n % 500 == 0:
            print(f"  {n}/{len(idx)}", flush=True)

    partner = {c: (conf[c].most_common(1)[0][0] if conf[c] else None) for c in classes}
    neg = []
    for i in idx:
        h = partner.get(meta[i]["taam"])
        if h:
            neg.append(srv.verify_match(np.array(X[i]), h)[1])

    pos, neg = np.array(pos), np.array(neg)
    say_wrong = float(np.percentile(pos, CONTRADICT_BUDGET * 100))
    say_ok = float(np.percentile(neg, (1 - BLESS_BUDGET) * 100))
    if say_ok < say_wrong:            # distributions barely overlap: no band needed
        say_ok = say_wrong

    contradicts = float((pos < say_wrong).mean())
    blesses = float((neg >= say_ok).mean())
    neutral_ok = float(((pos >= say_wrong) & (pos < say_ok)).mean())
    neutral_bad = float(((neg >= say_wrong) & (neg < say_ok)).mean())

    print(f"\n  say_wrong_below   = {say_wrong:.3f}")
    print(f"  say_correct_above = {say_ok:.3f}")
    print(f"\n  contradicts a correct reading   {contradicts*100:5.1f}%")
    print(f"  confirms a real mistake         {blesses*100:5.1f}%")
    print(f"  correct readings left neutral   {neutral_ok*100:5.1f}%")
    print(f"  mistakes caught as neutral      {neutral_bad*100:5.1f}%  "
          f"(shown grey, not blessed)")
    print(f"\n  mistakes NOT confirmed = {(1-blesses)*100:.1f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps({
        "say_wrong_below": round(say_wrong, 4),
        "say_correct_above": round(say_ok, 4),
        "contradicts_correct_rate": round(contradicts, 4),
        "confirms_real_mistake_rate": round(blesses, 4),
        "neutral_share_correct": round(neutral_ok, 4),
        "neutral_share_mistakes": round(neutral_bad, 4),
        "head": args.head or "production_verify_head.pth",
        "n_positives": len(pos), "n_negatives": len(neg),
        "held_out_parashot": sorted(held),
        "contradict_budget": CONTRADICT_BUDGET,
        "bless_budget": BLESS_BUDGET,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT / args.out}")


if __name__ == "__main__":
    main()
