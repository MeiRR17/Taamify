"""
Calibrate the verify head's decision threshold, and find an "unsure" band.
==========================================================================
`verify_match()` calls it a match at probability >= 0.5. That 0.5 was never
measured -- it is the default anyone writes first, and the product now shows
a child "you got it wrong" based on it.

The two errors are not equally expensive here. A MISS (the reading was wrong
and the system stayed quiet) costs one uncorrected mistake in a practice
session. A FALSE ALARM (the reading was right and the system said wrong)
costs trust: a learner who is contradicted while correct stops believing the
feedback, and after a couple of those stops using the app. So the operating
point should not maximise accuracy -- it should hold false alarms low and
accept misses.

This measures the full precision/recall curve on HELD-OUT PARASHOT (audio
the model never saw), and reports:

  * the threshold that keeps the false-alarm rate under a target
  * an "unsure" band around the boundary, where the honest answer is to say
    nothing rather than guess -- the app can then show a neutral mark
    instead of a verdict

Negatives are built the way the head was trained: the same word paired with
a wrong ta'am. Two kinds are reported separately, because they are not
equally hard -- a random wrong class, and the CONFUSABLE wrong class the
classifier itself most often mistakes it for. The second is the realistic
one: a learner's error is rarely random.

Usage:
  python Data_Prep/calibrate_verify_head.py
"""
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).resolve().parent.parent
FEAT = P / "features_v3_word"
OUT = P / "results" / "calibration"
TARGET_FA = 0.02          # at most 2% of correct readings may be called wrong
NEUTRAL_HARD = 0.15       # at most 15% of realistic mistakes may be called fine


def load_server(head_path=None):
    spec = importlib.util.spec_from_file_location("srv", P / "webapp" / "server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    if head_path:
        srv.VERIFY_HEAD_PATH = Path(head_path)
    return srv


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None)
    ap.add_argument("--tag", default="")
    args_cli = ap.parse_args()
    srv = load_server(args_cli.head)
    st = srv.get_model()
    classes = list(st["class_names"])

    X = np.load(FEAT / "BEN13R_X.npy", mmap_mode="r")
    meta = json.loads((FEAT / "BEN13R_meta.json").read_text(encoding="utf-8"))

    # the same 8 parashot the honest split holds out -- never trained on
    rng = np.random.default_rng(42)
    parashot = np.array([str(m.get("parasha", "?")) for m in meta])
    held = set(rng.choice(np.unique(parashot), size=8, replace=False).tolist())
    idx = [i for i, m in enumerate(meta)
           if str(m.get("parasha")) in held and m["taam"] in classes]
    rng.shuffle(idx)
    idx = idx[:3000]
    print(f"held-out parashot: {sorted(held)}")
    print(f"evaluating on {len(idx):,} words the model never trained on\n")

    # which class does the classifier most often confuse each class with?
    conf = defaultdict(Counter)
    probs_pos, probs_rand, probs_hard = [], [], []
    for n, i in enumerate(idx, 1):
        spec = np.array(X[i])
        true = meta[i]["taam"]
        top3, _ = srv.classify(spec)
        if top3 and top3[0][0] != true:
            conf[true][top3[0][0]] += 1
        _, p = srv.verify_match(spec, true)
        probs_pos.append(p)
        wrong = rng.choice([c for c in classes if c != true])
        _, p = srv.verify_match(spec, str(wrong))
        probs_rand.append(p)
        if n % 500 == 0:
            print(f"  {n}/{len(idx)}", flush=True)

    # second pass for the realistic negative: the class most confused with
    hard_for = {c: (conf[c].most_common(1)[0][0] if conf[c] else None) for c in classes}
    for i in idx[:1500]:
        true = meta[i]["taam"]
        h = hard_for.get(true)
        if not h:
            continue
        _, p = srv.verify_match(np.array(X[i]), h)
        probs_hard.append(p)

    pos = np.array(probs_pos)
    neg_r = np.array(probs_rand)
    neg_h = np.array(probs_hard)

    L = []
    def say(s=""):
        L.append(s); print(s)

    say("=" * 68)
    say("  Verify-head calibration — held-out parashot")
    say("=" * 68)
    say(f"\n  positives (word + its true ta'am):      n={len(pos):,}  "
        f"mean={pos.mean():.3f}")
    say(f"  negatives (word + a random wrong ta'am): n={len(neg_r):,}  "
        f"mean={neg_r.mean():.3f}")
    say(f"  negatives (word + its CONFUSABLE class): n={len(neg_h):,}  "
        f"mean={neg_h.mean():.3f}")
    say("\n  the third row is the realistic one: a learner's mistake is")
    say("  rarely a random ta'am, it is the one that sounds similar\n")

    # Column names say which error each number is, in product terms. An
    # earlier version printed the negatives-accepted rate under the heading
    # "false alarm" and (1-recall) under "miss" -- exactly backwards from the
    # definitions at the top of this file, and the threshold rule below
    # inherited the swap: it budgeted 2% for the cheap error and left the
    # expensive one unbounded, landing on 0.60, where 6% of correct readings
    # are contradicted.
    say(f"  {'thresh':>7}{'accepts-correct':>17}{'CONTRADICTS':>13}"
        f"{'lets-wrong-by':>15}{'(confusable)':>14}")
    say(f"  {'':>7}{'(recall)':>17}{'a correct kid':>13}{'random':>15}{'':>14}")
    say("  " + "-" * 66)
    rows = []
    for t in [.2, .3, .4, .5, .6, .7, .8, .9, .95]:
        recall = float((pos >= t).mean())
        contradicts = 1 - recall                     # correct reading called wrong
        lets_by_r = float((neg_r >= t).mean())       # wrong reading called fine
        lets_by_h = float((neg_h >= t).mean()) if len(neg_h) else float("nan")
        rows.append((t, recall, contradicts, lets_by_r, lets_by_h))
        mark = "  <- old default" if abs(t - .5) < 1e-9 else ""
        say(f"  {t:>7.2f}{recall:>17.3f}{contradicts:>13.3f}"
            f"{lets_by_r:>15.3f}{lets_by_h:>14.3f}{mark}")

    # The operating point is chosen on the EXPENSIVE error: at most TARGET_FA
    # of correct readings may ever be contradicted. Among thresholds that
    # respect that, take the one that lets fewest real (confusable) mistakes
    # through.
    ok = [r for r in rows if r[2] <= TARGET_FA]
    chosen = max(ok, key=lambda r: r[0]) if ok else min(rows, key=lambda r: r[2])
    say(f"\n  single-threshold operating point: {chosen[0]:.2f}")
    say(f"    contradicts {chosen[2]*100:.1f}% of correct readings "
        f"(budget <= {TARGET_FA*100:.0f}%)")
    say(f"    lets {chosen[4]*100:.1f}% of realistic (confusable) mistakes through")

    # Two thresholds are better than one here, because the two verdicts carry
    # different costs and so deserve different evidence. Below say_wrong: at
    # most TARGET_FA of correct readings live there, so "wrong" is safe to
    # say. Above say_correct: only NEUTRAL_HARD of realistic mistakes reach
    # it. Between them the honest answer is a neutral mark.
    say_wrong = float(np.percentile(pos, TARGET_FA * 100))
    say_correct = float(np.percentile(neg_h, 100 - NEUTRAL_HARD * 100)) if len(neg_h) else say_wrong
    if say_correct < say_wrong:
        say_correct = say_wrong
    neutral_pos = float(((pos >= say_wrong) & (pos < say_correct)).mean())
    neutral_hard = float(((neg_h >= say_wrong) & (neg_h < say_correct)).mean()) if len(neg_h) else 0.0
    say(f"\n  two-threshold contract (what the app should use):")
    say(f"    p <  {say_wrong:.3f}  -> say WRONG    "
        f"({TARGET_FA*100:.0f}% of correct readings land here — the trust cost)")
    say(f"    p >= {say_correct:.3f}  -> say CORRECT  "
        f"({NEUTRAL_HARD*100:.0f}% of real mistakes reach here — the miss cost)")
    say(f"    between      -> NEUTRAL, no verdict")
    say(f"      {neutral_pos*100:.1f}% of correct readings and "
        f"{neutral_hard*100:.1f}% of real mistakes fall in the neutral zone")
    say("\n" + "=" * 68)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report{args_cli.tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / f"scores{args_cli.tag}.npz",
                        pos=pos, neg_random=neg_r, neg_hard=neg_h)
    (OUT / f"calibration{args_cli.tag}.json").write_text(json.dumps({
        "say_wrong_below": round(say_wrong, 4),
        "say_correct_above": round(say_correct, 4),
        "single_threshold": round(chosen[0], 3),
        "contradicts_correct_rate": round(chosen[2], 4),
        "lets_confusable_mistake_through": round(chosen[4], 4),
        "neutral_share_correct": round(neutral_pos, 4),
        "neutral_share_mistakes": round(neutral_hard, 4),
        "n_positives": len(pos), "n_negatives": len(neg_r),
        "held_out_parashot": sorted(held),
        "target_false_alarm": TARGET_FA,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT}/calibration{args_cli.tag}.json")


if __name__ == "__main__":
    main()
