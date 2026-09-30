"""
How serious is this mistake? — the halachic severity model.
============================================================
Until now every channel reported a flat right/wrong, which quietly asserts
that all errors weigh the same. The sources say otherwise, and the
difference is large enough to change what the product should say.

THE RULE THE SOURCES CONVERGE ON

Peninei Halakha (ch. 15 §16, ch. 5 §13) lays out three positions and treats
the middle one as accepted practice: **an error that changes the MEANING is
corrected; one that does not is not.** Errors in letters or vocalisation
that change meaning send the reader back; an error in the ta'amim that does
not change meaning draws a rebuke but no repetition.

WHY THE TA'AMIM ARE NOT ALL EQUAL

The lenient treatment of ta'amim is often quoted as if it covered all of
them. It does not: the same source corrects a cantillation error that alters
how the verse divides, because that changes the sense. The example cited is
the placement of the etnachta in Exodus 17:9.

That distinction is already encoded in this project. taam_grammar's
DISJUNCTIVE_RANK says which ta'amim divide the verse and how strongly:

  rank 1-2 disjunctive  Siluk, Etnachta, Zaqef, Tipecha, Segol, Shalshelet
                        -- these carry the syntax. Get one wrong and the
                        verse divides somewhere else, which is a change of
                        meaning. STRICT.
  rank 3-4 disjunctive  smaller divisions. MODERATE.
  conjunctive           Mercha, Munach, Mahapakh, Darga, Qadma -- they join
                        rather than divide, so swapping one for another does
                        not move any boundary. LENIENT.

MEASURED, AND IT REFRAMES THE PROJECT'S HEADLINE NUMBER

On 2,500 held-out words the model scores 90.8% on strong disjunctives,
93.9% on weak ones, and 80.2% on conjunctives. Only 19% of all its errors
touch a ta'am that changes meaning. The 86.9% that has been quoted all along
averages over a distinction that matters: accuracy on the ta'amim that can
actually invalidate a reading is far higher than accuracy on the ones that
cannot.

THE OTHER CHANNELS ARE THE STRICT ONES

Stress is not a stylistic matter: בָּאָה mil'el is past tense and milra is
present; אוֹרִי milra is "my light" and mil'el is a feminine imperative.
Those are meaning changes, so the stress channel is strict. So is the
ayin/alef distinction, which swaps the root. This inverts the project's
instinct: **be forgiving about melody and strict about grammar.**

NOT A HALACHIC RULING. This encodes a reading of accessible sources in order
to rank feedback severity in a practice app. Anything touching whether an
obligation was actually discharged needs a rabbi.

Usage:
  python Data_Prep/severity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taam_grammar as tg   # noqa: E402

# how the product should speak about each level
STRICT = "strict"        # changes meaning -> a real error
MODERATE = "moderate"    # may change meaning -> worth flagging clearly
LENIENT = "lenient"      # does not change meaning -> a note, not an error

LEVEL_HE = {
    STRICT: "משנה משמעות",
    MODERATE: "עלול לשנות משמעות",
    LENIENT: "אינו משנה משמעות",
}

# per channel. The ta'am channel is resolved per-class below.
#
# stress and guttural are downgraded from their grammatically-correct STRICT
# rank (see the comments below) on 2026-09-30, pending calibration. Measured
# through the live API on Ben13's own CORRECT continuous readings, these
# channels call "wrong" on audio that is right at rates the ta'am channel
# would never be allowed to ship at:
#
#   stress    17.0% false "wrong"   (ta'am's calibrated rate: 2.0%)
#   dagesh    18.4%
#   guttural  10.6%
#   sheva     20.4%
#
# Unlike the ta'am channel, these run_channels() verdicts are a raw argmax
# with no window-geometry tuning, no align-quality gate, and no percentile-
# derived threshold -- exactly the three fixes that took the ta'am channel's
# own false-contradict rate from ~14% down to 2%, none of which have been
# done here yet. A confidence cutoff only partly separates true from false
# (median confidence 0.99 right vs 0.86-0.94 wrong -- much closer than
# ta'am's pattern), so it is not a substitute for that work.
#
# STRICT drives should_repeat: a learner is told the reading is invalid and
# must be redone. That is not a claim these two channels can currently back
# up, so they are held at MODERATE (a note, not a repeat) until they get the
# same treatment ta'am received. dagesh/sheva were already MODERATE, so no
# change there, but their false-positive rates are just as high and their
# messages should be read with that caveat until the same work is done.
CHANNEL_SEVERITY = {
    "stress": MODERATE,    # STRICT in principle (בָּאָה/אוֹרִי) -- see above
    "guttural": MODERATE,  # STRICT in principle (ayin/alef swaps the root)
    "dagesh": MODERATE,    # b/v, k/kh can change a word, often does not
    "sheva": MODERATE,     # pronounced or not; can shift a form
}

# Below this per-word confidence, run_channels() reports "unsure" instead of
# a verdict either way -- the same three-state discipline the ta'am channel
# uses, applied crudely here since a proper percentile calibration has not
# been done. Halves the false-"wrong" rate at the cost of ~15% coverage; see
# the comment above CHANNEL_SEVERITY for the full numbers.
CHANNEL_MIN_CONFIDENCE = 0.9


def taam_severity(taam: str) -> str:
    """Strong dividers change where the verse breaks, and so change sense.
    Servants join rather than divide, and swapping one for another moves no
    boundary at all."""
    rank = tg.DISJUNCTIVE_RANK.get(taam)
    if rank in (1, 2):
        return STRICT
    if rank in (3, 4):
        return MODERATE
    return LENIENT


def verdict_for(channel: str, taam: str = None, expected=None, heard=None):
    """A severity plus the sentence the product should actually say."""
    if channel == "taam":
        level = taam_severity(expected or taam)
        he_exp = tg.TAAM_HE.get(expected, expected)
        he_heard = tg.TAAM_HE.get(heard, heard)
        if level == LENIENT:
            msg = (f"כאן המסורת שרה {he_exp}, ונשמע {he_heard} — "
                   f"שניהם משרתים, אז המשמעות לא משתנה")
        elif level == MODERATE:
            msg = f"צריך {he_exp}, נשמע {he_heard} — זה מזיז את חלוקת הפסוק"
        else:
            msg = (f"צריך {he_exp}, נשמע {he_heard} — "
                   f"זה מחלק את הפסוק במקום אחר ומשנה את המשמעות")
        return {"level": level, "level_he": LEVEL_HE[level], "message": msg}

    level = CHANNEL_SEVERITY.get(channel, MODERATE)
    msgs = {
        "stress": "ההטעמה במקום הלא נכון — כאן זה משנה את משמעות המילה",
        "guttural": "צריך ע׳ גרונית — היא מחליפה את שורש המילה",
        "dagesh": "הדגש הקל לא נשמע — ב׳ מול ו׳, כ׳ מול כ׳ רפה",
        "sheva": "השווא — כאן הוא אמור להישמע (או לא) אחרת",
    }
    return {"level": level, "level_he": LEVEL_HE[level],
            "message": msgs.get(channel, "")}


def summarise(word_results):
    """Roll per-word channel verdicts into one severity for the phrase, so the
    app can say 'this needs repeating' versus 'worth noticing'."""
    worst = LENIENT
    order = {LENIENT: 0, MODERATE: 1, STRICT: 2}
    counts = {STRICT: 0, MODERATE: 0, LENIENT: 0}
    for w in word_results:
        for v in (w.get("severities") or []):
            counts[v] += 1
            if order[v] > order[worst]:
                worst = v
    return {"worst": worst, "worst_he": LEVEL_HE[worst], "counts": counts,
            "should_repeat": worst == STRICT}


if __name__ == "__main__":
    print("חומרה לפי טעם:\n")
    rows = []
    for t in sorted(set(tg.DISJUNCTIVE_RANK) | set(tg.CONJUNCTIVE)):
        rows.append((taam_severity(t), tg.TAAM_HE.get(t, t), t,
                     tg.DISJUNCTIVE_RANK.get(t)))
    for level in (STRICT, MODERATE, LENIENT):
        sel = [r for r in rows if r[0] == level]
        print(f"  {LEVEL_HE[level]}  ({len(sel)}):")
        print("    " + " · ".join(r[1] for r in sel))
        print()
    print("חומרה לפי ערוץ:")
    for ch, lv in CHANNEL_SEVERITY.items():
        print(f"  {ch:<10} {LEVEL_HE[lv]}")
