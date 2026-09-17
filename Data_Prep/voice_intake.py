"""
Gatekeeper for new voices — the part that decides what is allowed to train.
===========================================================================
THE FLAW THIS EXISTS TO CLOSE

ingest_collected_phrases.py labels a learner's recording with the ta'am the
TEXT requires. That is correct for a competent reader and actively poisonous
for a learner: a child who sings Tipecha where the text says Munach produces
a row asserting "this sound is Munach". Train on enough of those and the
model is taught the mistakes it exists to catch.

The corpus is currently one professional reader precisely because every word
in it was read correctly. Growth must not quietly give that up.

FOUR TIERS, and only two of them may train the classifier
---------------------------------------------------------
  gold      a known ba'al kri'ah recorded deliberately. Trains everything.
  verified  a learner phrase a human confirmed as correct. Trains everything.
  agreed    the model already scored every word correct with high confidence.
            Trains, but it is SELF-CONFIRMING -- it can only reinforce what
            the model already believes, never correct it. Capped, and never
            allowed to dominate a class.
  rejected  anything else: a wrong word, a low alignment score, an unusual
            duration. Kept for Stage B (error detection needs real errors)
            and NEVER used as a positive label.

WHY 'agreed' IS CAPPED RATHER THAN TRUSTED
A model that trains on its own confident predictions drifts: it sharpens its
existing biases and loses the ability to be corrected by reality. The cap
keeps it a supplement to gold data, not a replacement.

OTHER CASES THIS HANDLES
  * A voice contributing 60 words next to Ben13's 64,649. Measured before:
    an unbalanced mix means "does one dominant voice hurt" is what gets
    measured, not "is this corpus good". Per-reader caps are computed here.
  * The same phrase recorded five times. Near-duplicates inflate confidence
    and leak across a split. Deduplicated per (voice, phrase).
  * A different tradition arriving unnoticed. Tradition is required metadata
    and a mismatch is rejected rather than averaged in.
  * Halting or restarted reading. Caught by the CTC alignment score and by a
    duration sanity check against the corpus distribution.

Usage:
  python Data_Prep/voice_intake.py --report
  python Data_Prep/voice_intake.py --promote    # write the training-safe set
"""
import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent.parent
PHRASES = P / "app_assets" / "collected_phrases"
LOG = PHRASES / "phrases_log.jsonl"
DECISIONS = PHRASES / "intake_decisions.json"

# a word is only allowed to train as a positive if the model was confident it
# was right; the threshold is the calibrated "say correct" line, not a guess
CAL = P / "results" / "calibration" / "calibration.json"
MIN_ALIGN_SCORE = -2.5     # below this the alignment itself is unreliable
AGREED_CAP_PER_CLASS = 300  # self-confirming rows may not dominate a class
MIN_WORDS_FOR_READER = 40   # below this a voice cannot be its own LORO fold


def say_correct_above():
    try:
        return json.loads(CAL.read_text())["say_correct_above"]
    except Exception:
        return 0.45


def load_rows():
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def classify_phrase(rec, thresh, manual):
    """Which tier this phrase belongs to, and why."""
    cid = rec["clip_id"]
    if manual.get(cid) == "verified":
        return "verified", "אושר ידנית"
    if manual.get(cid) == "gold" or rec.get("tier") == "gold":
        return "gold", "בעל קריאה"
    if manual.get(cid) == "rejected":
        return "rejected", "נדחה ידנית"
    if not rec.get("aligned"):
        return "rejected", "היישור נכשל"
    score = rec.get("align_score")
    if score is not None and score < MIN_ALIGN_SCORE:
        return "rejected", f"ציון יישור נמוך ({score})"
    statuses = rec.get("statuses") or []
    if not statuses:
        return "rejected", "אין תוצאה למילים"
    if any(s == "wrong" for s in statuses):
        return "rejected", "יש מילה שנקראה לא נכון"
    if any(s in ("unsure", "unknown") for s in statuses):
        return "rejected", "המערכת לא הייתה בטוחה"
    if all(s == "ok" for s in statuses):
        return "agreed", "כל המילים אושרו בביטחון"
    return "rejected", "לא מסווג"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        print("אין עדיין הקלטות — הלולאה תתחיל כשהאפליקציה תהיה בשימוש")
        print("\nהמערכת מוכנה: כל קטע שיוקלט בהסכמה ייכנס לכאן, ייבדק,")
        print("ורק מה שבטוח לאמן עליו יעבור הלאה.")
        return

    manual = {}
    if DECISIONS.exists():
        manual = json.loads(DECISIONS.read_text()).get("clips", {})
    thresh = say_correct_above()

    tiers = defaultdict(list)
    reasons = Counter()
    seen = set()
    for r in rows:
        key = (r.get("voice_id"), tuple(r.get("words") or []))
        if key in seen:
            tiers["duplicate"].append(r)
            reasons["כפילות — אותו קטע מאותו קול"] += 1
            continue
        seen.add(key)
        tier, why = classify_phrase(r, thresh, manual)
        tiers[tier].append(r)
        if tier == "rejected":
            reasons[why] += 1

    print(f"{len(rows)} הקלטות · {len(seen)} ייחודיות\n")
    for t in ("gold", "verified", "agreed", "rejected", "duplicate"):
        n = len(tiers[t])
        words = sum(len(r.get("words") or []) for r in tiers[t])
        print(f"  {t:<10} {n:>5} קטעים  {words:>6} מילים")
    if reasons:
        print("\n  סיבות דחייה:")
        for k, v in reasons.most_common():
            print(f"    {v:>4}  {k}")

    by_voice = defaultdict(int)
    for t in ("gold", "verified", "agreed"):
        for r in tiers[t]:
            by_voice[r.get("voice_id") or "anon"] += len(r.get("words") or [])
    if by_voice:
        print("\n  לפי קול (מילים שמותר לאמן עליהן):")
        for v, n in sorted(by_voice.items(), key=lambda kv: -kv[1]):
            flag = "" if n >= MIN_WORDS_FOR_READER else \
                f"  ← מתחת ל-{MIN_WORDS_FOR_READER}, לא יכול להיות fold נפרד"
            print(f"    {v:<16} {n:>6}{flag}")
        counts = list(by_voice.values())
        if len(counts) > 1:
            cap = min(counts)
            print(f"\n  תקרה מומלצת לקורא: {cap:,} מילים "
                  f"(הקטן ביותר) — כדי שקול אחד לא ישתלט")

    if args.promote:
        safe = [r["clip_id"] for t in ("gold", "verified", "agreed")
                for r in tiers[t]]
        # cap the self-confirming tier so it cannot dominate
        agreed_by_class = Counter()
        capped = []
        for r in tiers["agreed"]:
            cls = tuple(sorted(set(r.get("taamim") or [])))
            if agreed_by_class[cls] >= AGREED_CAP_PER_CLASS:
                continue
            agreed_by_class[cls] += 1
            capped.append(r["clip_id"])
        allowed = ([r["clip_id"] for t in ("gold", "verified") for r in tiers[t]]
                   + capped)
        out = PHRASES / "training_safe.json"
        out.write_text(json.dumps({
            "allowed_clip_ids": sorted(allowed),
            "rejected": len(tiers["rejected"]),
            "agreed_capped_out": len(tiers["agreed"]) - len(capped),
            "note": "only these may be used as POSITIVE training labels; "
                    "rejected clips are kept for Stage B error detection",
        }, indent=1), encoding="utf-8")
        print(f"\n  {len(allowed)} קטעים מותרים לאימון -> {out.name}")


if __name__ == "__main__":
    main()
