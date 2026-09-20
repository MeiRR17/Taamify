"""
Calibrate the thresholds against the score the server actually serves.
=======================================================================
THE BUG THIS EXISTS TO CLOSE

derive_neutral_band.py derived say_wrong_below / say_correct_above from
verify_match() -- the isolated-word score. /api/check_phrase now acts on the
context-blended score, which has a different distribution. Thresholds derived
against one score and applied to another are simply wrong, and nothing would
have failed loudly: the endpoint would keep answering, just at the wrong
operating point.

So this derives the thresholds along the same path the request takes:
reconstruct the verse, run the sequence tagger over it, blend, and only then
take percentiles.

THE TWO BUDGETS, UNCHANGED

  say_wrong_below    at most CONTRADICT_BUDGET of correct readings may be
                     called wrong. This is the expensive error: a learner
                     told they were wrong when they were right stops
                     believing the system, and there is no recovering from
                     that.
  say_correct_above  at most BLESS_BUDGET of real mistakes may be confirmed
                     as correct.

Between them is the neutral band. If the negatives separate well enough that
one threshold satisfies both budgets, the band is empty -- which is a result,
not a failure, and it is reported as such.

THE NEGATIVES ARE HARD ONES

A random wrong ta'am is easy to reject and teaches the threshold nothing. The
negative used here is the ta'am the model itself most nearly confused this
word with, which is the mistake a learner is actually likely to make.

Usage:
  python Data_Prep/calibrate_served_path.py
  python Data_Prep/calibrate_served_path.py --verses 400 --out calibration.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import eval_context_verify as ev

P = Path(__file__).resolve().parent.parent
CONTRADICT_BUDGET = 0.02
BLESS_BUDGET = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", type=int, default=350)
    ap.add_argument("--out", default="calibration_served.json")
    args = ap.parse_args()

    srv = ev.load_server()
    X = np.load(P / "features_v3_word" / "BEN13R_X.npy", mmap_mode="r")
    meta = json.loads((P / "features_v3_word" / "BEN13R_meta.json")
                      .read_text(encoding="utf-8"))
    classes = set(srv.get_model()["class_names"])

    rng = np.random.default_rng(42)
    par = np.unique([str(m.get("parasha", "?")) for m in meta])
    held = set(rng.choice(par, size=8, replace=False).tolist())
    verses = ev.build_verses(meta, classes, held, args.verses, rng)
    print(f"  {len(verses)} פסוקים מ-{len(held)} פרשות מוחזקות")
    if not verses:
        print("  לא שוחזר אף פסוק — אין מה לכייל.")
        return

    pos, neg, pos_cls = [], [], []
    for ref, idxs in verses:
        specs = [np.array(X[i]) for i in idxs]
        ctx = srv.context_probabilities(specs)
        for k, i in enumerate(idxs):
            true_t = meta[i]["taam"]
            cp = ctx[k].get(true_t) if ctx else None
            conf, _ = srv.score_word(specs[k], true_t, cp)
            pos.append(conf)
            pos_cls.append(true_t)
            # the hard negative: what the model most nearly heard instead
            top3, _ = srv.classify(specs[k])
            alt = next((t for t, _ in (top3 or []) if t != true_t), None)
            if alt is None:
                continue
            acp = ctx[k].get(alt) if ctx else None
            nconf, _ = srv.score_word(specs[k], alt, acp)
            neg.append(nconf)

    pos, neg = np.array(pos), np.array(neg)
    print(f"  {len(pos):,} קריאות נכונות · {len(neg):,} שליליות קשות\n")

    say_wrong = float(np.percentile(pos, CONTRADICT_BUDGET * 100))
    say_ok = float(np.percentile(neg, (1 - BLESS_BUDGET) * 100))
    band = say_ok > say_wrong
    if not band:
        say_ok = say_wrong

    contradicts = float((pos < say_wrong).mean())
    blesses = float((neg >= say_ok).mean())
    neutral_ok = float(((pos >= say_wrong) & (pos < say_ok)).mean())
    neutral_bad = float(((neg >= say_wrong) & (neg < say_ok)).mean())
    caught = float((neg < say_wrong).mean())

    print(f"  say_wrong_below   = {say_wrong:.3f}")
    print(f"  say_correct_above = {say_ok:.3f}")
    print(f"  רצועה ניטרלית: {'קיימת' if band else 'ריקה'}\n")
    print(f"  סותר קריאה נכונה        {contradicts*100:5.1f}%   "
          f"(תקציב {CONTRADICT_BUDGET*100:.0f}%)")
    print(f"  מאשר טעות אמיתית        {blesses*100:5.1f}%   "
          f"(תקציב {BLESS_BUDGET*100:.0f}%)")
    print(f"  תופס טעות אמיתית        {caught*100:5.1f}%")
    if band:
        print(f"  נכונות שנשארו אפורות    {neutral_ok*100:5.1f}%")
        print(f"  טעויות שנשארו אפורות    {neutral_bad*100:5.1f}%")
    else:
        print("\n  שני התקציבים מסתפקים בסף אחד — הרצועה ריקה. "
              "המצב האפור במוצר נשאר ליישור שנכשל, לא לחוסר ביטחון.")

    out = P / "results" / "calibration" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "say_wrong_below": round(say_wrong, 4),
        "say_correct_above": round(say_ok, 4),
        "has_neutral_band": band,
        "contradict_rate": round(contradicts, 4),
        "bless_rate": round(blesses, 4),
        "catch_rate": round(caught, 4),
        "contradict_budget": CONTRADICT_BUDGET,
        "bless_budget": BLESS_BUDGET,
        "context_weight": srv.CONTEXT_WEIGHT,
        "n_positive": len(pos), "n_negative": len(neg),
        "n_verses": len(verses),
        "held_out_parashot": sorted(held),
        "derived_on": ("the context-blended score the server serves, "
                       "not the isolated-word score"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  נשמר: {out}")


if __name__ == "__main__":
    main()
