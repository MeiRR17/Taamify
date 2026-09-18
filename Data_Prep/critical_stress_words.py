"""
Where stress carries meaning: the critical-words list.
======================================================
The severity model says stress errors are strict, because בָּאָה mil'el is
past tense and milra is present, and אוֹרִי milra is "my light" while mil'el
is a feminine imperative. Those are two examples. The product needs the list.

HOW IT IS DERIVED

A word whose consonant-and-vowel skeleton appears in the Torah sometimes with
final stress and sometimes with penultimate stress is a word where the stress
is doing work -- it is the only thing separating the two readings. Counting
those over the whole Torah yields the list without a morphological analyser.

THE TRAP, AND WHY AN EARLIER RUN OF THIS WAS WRONG

stressed_syllable() reports which rule it used. Two of them --
`postpositive_single` and `prepositive_single` -- are not readings of the
text but fallbacks: a postpositive ta'am (Pashta, Zarqa, Telisha Qetana) is
written on the last letter of the word no matter where the stress falls, and
a prepositive (Telisha Gedola, Yetiv) on the first. The Masoretes repeat the
mark on the stressed syllable when it differs, so a DOUBLED mark is evidence
and a single one is not.

A first pass ignored that and reported וַיֹּאמֶר as a stress minimal pair
(560 mil'el vs 53 milra) -- which is nonsense: it is mil'el every time, and
the 53 were simply occurrences carrying a postpositive ta'am. Same for יהוה,
הארץ, and most of the list. Only the rules that actually locate the stress
are trusted here:

  taam_position        the mark sits on the stressed syllable -- direct
  meteg_silluq         verse-final, meteg marks the stress -- direct
  *_double             the repeated mark resolves it -- direct

TWO MORE FILTERS

A minority reading under 5 occurrences, or under 10% of the word's uses, is
far more likely to be an encoding artefact than a real alternation, so it is
dropped. What survives is small and checkable by eye.

Usage:
  python Data_Prep/critical_stress_words.py
"""
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from stressed_syllable import stressed_syllable   # noqa: E402
import taam_grammar as tg                         # noqa: E402

BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]
TRUSTED = {"taam_position", "meteg_silluq",
           "postpositive_double", "prepositive_double"}
MIN_MINORITY = 5
MIN_MINORITY_SHARE = 0.10

NIQQUD = set("ְֱֲֳִֵֶַָ"
             "ׇֹֺֻּׁׂ")


def skeleton(word):
    """Letters plus vowels, ta'amim stripped -- the form the reader sees once
    the cantillation is taken away. Two words sharing it differ ONLY in where
    the stress falls."""
    w = unicodedata.normalize("NFD", word)
    return "".join(c for c in w
                   if ("א" <= c <= "ת") or c in NIQQUD)


def main():
    cache = P / "Data_Prep" / "sefaria_cache"
    by_form = defaultdict(Counter)      # skeleton -> {from_end: count}
    examples = defaultdict(dict)        # skeleton -> {from_end: (word, ref)}
    n_words = n_trusted = 0

    for book in BOOKS:
        chap = 1
        while (cache / f"{book}.{chap}.json").exists():
            verses = json.loads((cache / f"{book}.{chap}.json")
                                .read_text(encoding="utf-8"))
            for vi, raw in enumerate(verses, start=1):
                for word, taam in tg.verse_tokens(raw):
                    if not taam:
                        continue
                    n_words += 1
                    r = stressed_syllable(word)
                    if not r or not r.get("ok"):
                        continue
                    if r.get("rule") not in TRUSTED:
                        continue
                    if r["n_syllables"] < 2:
                        continue
                    fe = r["from_end"]
                    if fe not in (1, 2):
                        continue
                    n_trusted += 1
                    sk = skeleton(word)
                    by_form[sk][fe] += 1
                    examples[sk].setdefault(fe, (word, f"{book} {chap}:{vi}"))
            chap += 1

    crit = {}
    for sk, counts in by_form.items():
        if len(counts) < 2:
            continue
        total = sum(counts.values())
        minority = min(counts.values())
        if minority < MIN_MINORITY:
            continue
        if minority / total < MIN_MINORITY_SHARE:
            continue
        crit[sk] = {
            "milra": counts.get(1, 0),
            "milel": counts.get(2, 0),
            "total": total,
            "minority_share": round(minority / total, 3),
            "example_milra": examples[sk].get(1),
            "example_milel": examples[sk].get(2),
        }

    print(f"  {n_words:,} מילים בתורה, {n_trusted:,} עם הטעמה ודאית "
          f"({n_trusted/n_words*100:.0f}%)")
    print(f"  {len(by_form):,} צורות ייחודיות")
    print(f"  **{len(crit)} צורות שבהן ההטעמה לבדה מבחינה בין שתי קריאות**\n")
    print(f"  {'צורה':<16}{'מלרע':<8}{'מלעיל':<8}{'מיעוט':<8}דוגמאות")
    for sk, d in sorted(crit.items(), key=lambda kv: -kv[1]["total"])[:30]:
        a = d["example_milra"][0] if d["example_milra"] else "—"
        b = d["example_milel"][0] if d["example_milel"] else "—"
        print(f"  {sk:<16}{d['milra']:<8}{d['milel']:<8}"
              f"{d['minority_share']*100:<8.0f}{a}  /  {b}")

    out = P / "results" / "melody_research" / "critical_stress_words.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": ("forms appearing in the Torah with BOTH final and "
                 "penultimate stress, counted only where the pointing "
                 "actually locates the stress"),
        "trusted_rules": sorted(TRUSTED),
        "min_minority": MIN_MINORITY,
        "min_minority_share": MIN_MINORITY_SHARE,
        "n_forms": len(crit),
        "words": crit,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  נשמר: {out}")


if __name__ == "__main__":
    main()
