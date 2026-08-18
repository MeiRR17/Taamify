"""
Locate the stressed syllable of a cantillated Hebrew word.
==========================================================
A ta'am is realized melodically on the *stressed* syllable, not spread evenly
over the word. Feeding the classifier a whole word therefore feeds it mostly
audio that carries no ta'am — for a 4-syllable word, three quarters of the
window is filler.

The vocalized text already says where the stress is; no acoustic model has to
guess it. In the Masoretic encoding the ta'am mark sits on the consonant that
opens the stressed syllable — with two families of exceptions:

  postpositive  (Segolta, Zarqa, Pashta, Telisha Qetana)
      always written on the *last* letter of the word.
  prepositive   (Telisha Gedola, Yetiv, Dehi)
      always written on the *first* letter of the word.

For those, the Masoretes *repeat* the mark on the stressed syllable whenever
the stress is not where the mark lands, so a doubled postpositive means the
first occurrence is the real stress; a doubled prepositive means the second is.
When it appears only once, the word is stressed at the mark's own edge.

Silluq is encoded with the meteg codepoint (U+05BD) and sits on the stressed
syllable, so verse-final words get their stress from the meteg.

Anything still unresolved falls back to final stress (milra), which is the
majority pattern in Biblical Hebrew.

Self-test:
  python Data_Prep/stressed_syllable.py
Corpus statistics (how often each rule fires):
  python Data_Prep/stressed_syllable.py --corpus aligned_dataset_pt.json
"""

import argparse
import json
import unicodedata
from collections import Counter

# --- Unicode inventory -------------------------------------------------------
LETTERS = set("אבגדהוזחטיכךלמנןסעפףצץקרשת")

VOWELS = set("ִֵֶַָׇֹֺֻ")  # hiriq..qubuts, qamats qatan
HATAFS = set("ֱֲֳ")                                       # hataf segol/patah/qamats
SHEVA = "ְ"
DAGESH = "ּ"
METEG = "ֽ"
SHIN_DOTS = set("ׁׂ")
RAFE = "ֿ"

POSTPOSITIVE = {
    "֒",  # Segolta
    "֘",  # Zarqa / Tsinnorit
    "֮",  # Zinor (prose zarqa)
    "֙",  # Pashta
    "֩",  # Telisha Qetana
}
PREPOSITIVE = {
    "֠",  # Telisha Gedola
    "֚",  # Yetiv
    "֭",  # Dehi
}
TAAMIM = set(chr(c) for c in range(0x0591, 0x05AF)) - {"֢", "֫", "֬"}

NIQQUD = VOWELS | HATAFS | {SHEVA, DAGESH, METEG, RAFE} | SHIN_DOTS


def parse_word(word: str) -> list:
    """Split into letter units: [{'letter', 'marks', 'idx'}], idx into the NFC string."""
    word = unicodedata.normalize("NFC", word)
    units = []
    for i, ch in enumerate(word):
        if ch in LETTERS:
            units.append({"letter": ch, "marks": "", "idx": i})
        elif units and (ch in NIQQUD or ch in TAAMIM):
            units[-1]["marks"] += ch
    return units


GUTTURALS = set("חעה")


def has_vowel(marks: str) -> bool:
    return any(m in VOWELS for m in marks) or any(m in HATAFS for m in marks)


def is_mater_vowel(units: list, k: int) -> bool:
    """Is unit k a vav carrying shuruq/holam-male for the *previous* consonant?

    Shuruq (וּ) and holam male (וֹ) are full vowels written on a vav. The vav is
    then a vowel sign, not a consonant: it belongs to the preceding letter and
    opens nothing. It is only a consonant when the letter before it already has
    a vowel of its own (then vav+dagesh is a doubled vav, vav+holam a vav with
    holam).
    """
    if k == 0 or units[k]["letter"] != "ו":
        return False
    marks = units[k]["marks"]
    if has_vowel(marks.replace("ֹ", "")) or SHEVA in marks:
        return False
    if DAGESH not in marks and "ֹ" not in marks:
        return False
    prev = units[k - 1]["marks"]
    return not has_vowel(prev) and SHEVA not in prev


def is_furtive_patah(units: list, k: int) -> bool:
    """Word-final guttural whose patah is pronounced before it (patah genuva).

    רוּחַ is one syllable, not two: the patah is an off-glide into the ḥet, so it
    opens no syllable and can never carry the stress.
    """
    if k != len(units) - 1 or k == 0:
        return False
    if units[k]["letter"] not in GUTTURALS:
        return False
    marks = units[k]["marks"]
    vowels = [m for m in marks if m in VOWELS or m in HATAFS]
    return vowels == ["ַ"]


def is_syllable_onset(units: list, k: int) -> bool:
    """Does unit k begin a new syllable?

    A letter opens a syllable when it carries a real vowel, a hataf, a vocal
    sheva, or a following mater (shuruq / holam male). Silent sheva, bare
    matres lectionis and a furtive patah close the previous syllable instead.
    """
    if k == 0:
        return True
    if is_mater_vowel(units, k):
        return False
    if is_furtive_patah(units, k):
        return False
    marks = units[k]["marks"]
    if has_vowel(marks):
        return True
    # A letter with no vowel of its own still opens a syllable when the next
    # unit is its mater — that is where shuruq and holam male live.
    if k + 1 < len(units) and is_mater_vowel(units, k + 1) and SHEVA not in marks:
        return True
    if SHEVA in marks:
        # A sheva is vocal after another sheva, or under a doubling dagesh.
        prev_marks = units[k - 1]["marks"]
        if SHEVA in prev_marks:
            return True
        if DAGESH in marks and has_vowel(prev_marks):
            return True
        return False
    return False


def syllables(units: list) -> list:
    """[(start_unit, end_unit_exclusive)] over the letter units."""
    onsets = [k for k in range(len(units)) if is_syllable_onset(units, k)]
    if not onsets:
        return [(0, len(units))]
    return [(onsets[i], onsets[i + 1] if i + 1 < len(onsets) else len(units))
            for i in range(len(onsets))]


def stressed_unit(units: list) -> tuple:
    """(unit index bearing the stress, rule name)."""
    positions = [(k, m) for k, u in enumerate(units) for m in u["marks"] if m in TAAMIM]

    if positions:
        marks_seen = [m for _, m in positions]
        for mark in set(marks_seen):
            if marks_seen.count(mark) < 2:
                continue
            hits = [k for k, m in positions if m == mark]
            if mark in POSTPOSITIVE:
                return hits[0], "postpositive_doubled"
            if mark in PREPOSITIVE:
                return hits[-1], "prepositive_doubled"

        # Single occurrence: a positional mark tells us nothing beyond its edge.
        non_positional = [(k, m) for k, m in positions
                          if m not in POSTPOSITIVE and m not in PREPOSITIVE]
        if non_positional:
            return non_positional[-1][0], "taam_position"
        mark = positions[0][1]
        if mark in POSTPOSITIVE:
            return positions[0][0], "postpositive_single"
        return positions[0][0], "prepositive_single"

    # Verse-final words carry silluq, encoded as meteg, on the stressed syllable.
    metegs = [k for k, u in enumerate(units) if METEG in u["marks"]]
    if metegs:
        return metegs[-1], "meteg_silluq"

    return len(units) - 1, "fallback_milra"


def stressed_syllable(word: str) -> dict:
    """Locate the stressed syllable.

    Returns index/span of the stressed syllable, its text, and which rule fired.
    `char_start`/`char_end` index into the NFC-normalized input string, so the
    span can be mapped onto a character-level acoustic alignment.
    """
    units = parse_word(word)
    if not units:
        return {"ok": False, "reason": "no letters"}

    syls = syllables(units)
    unit_idx, rule = stressed_unit(units)
    # The mark sits on the syllable's onset consonant; if it landed mid-syllable
    # (a coda letter), the syllable containing it is still the right one.
    syl_idx = max(i for i, (s, _e) in enumerate(syls) if s <= unit_idx)
    s, e = syls[syl_idx]

    nfc = unicodedata.normalize("NFC", word)
    char_start = units[s]["idx"]
    char_end = units[e]["idx"] if e < len(units) else len(nfc)

    return {
        "ok": True,
        "n_syllables": len(syls),
        "stress_index": syl_idx,
        "from_end": len(syls) - syl_idx,      # 1 = milra, 2 = mil'el
        "text": nfc[char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
        # Indices into the bare consonant sequence (== positions in word_clean),
        # which is what a character-level CTC alignment emits.
        "letter_start": s,
        "letter_end": e,
        "n_letters": len(units),
        "rule": rule,
    }


# --- checks ------------------------------------------------------------------
CASES = [
    # (word, expected stressed syllable consonants, note)
    ("בְּרֵאשִׁ֖ית", "שׁ", "Tipeha on the stressed syllable"),
    ("בָּרָ֣א", "רָ", "Munah, milra"),
    ("אֱלֹהִ֑ים", "הִ", "Etnahta, milra"),
    ("הָאָֽרֶץ", "אָ", "silluq via meteg, mil'el"),
    ("וַיֹּ֥אמֶר", "יֹּ", "Merkha, mil'el"),
]


def self_test():
    print(f"{'word':<16}{'syl':>4}{'stress':>7}{'from-end':>9}  {'rule':<22}{'got':<10}{'want'}")
    print("-" * 78)
    ok = 0
    for word, want, note in CASES:
        r = stressed_syllable(word)
        got = "".join(c for c in r["text"] if c in LETTERS or c in VOWELS or c == DAGESH)
        got_letters = "".join(c for c in r["text"] if c in LETTERS)
        hit = want.strip()[0] in got_letters
        ok += hit
        print(f"{word:<16}{r['n_syllables']:>4}{r['stress_index']:>7}{r['from_end']:>9}  "
              f"{r['rule']:<22}{r['text']:<10}{want}  {'OK' if hit else 'MISS'}")
    print("-" * 78)
    print(f"{ok}/{len(CASES)} cases hit the expected syllable\n")


def corpus_stats(path: str):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    records = [r for r in records if r.get("word_raw")]
    rules, from_end, per_taam = Counter(), Counter(), {}
    syl_counts = Counter()
    for r in records:
        res = stressed_syllable(r["word_raw"])
        if not res["ok"]:
            rules["unparseable"] += 1
            continue
        rules[res["rule"]] += 1
        from_end[res["from_end"]] += 1
        syl_counts[res["n_syllables"]] += 1
        per_taam.setdefault(r["taam"], Counter())[res["rule"]] += 1

    n = sum(rules.values())
    print(f"words analysed: {n:,}\n")
    print("rule that determined the stress:")
    for k, v in rules.most_common():
        print(f"  {k:<24}{v:>8,}  {100 * v / n:>5.1f}%")
    print("\nstress position (1 = milra / final):")
    for k, v in sorted(from_end.items()):
        print(f"  {k:<24}{v:>8,}  {100 * v / n:>5.1f}%")
    print("\nsyllables per word:")
    for k, v in sorted(syl_counts.items()):
        print(f"  {k:<24}{v:>8,}  {100 * v / n:>5.1f}%")
    print("\npositional ta'amim — how often the doubling rule resolved the stress:")
    for taam in ("Pashta", "Segol", "Zarqa", "Telisha_Qetana", "Telisha_Gedola", "Yetiv"):
        c = per_taam.get(taam)
        if not c:
            continue
        tot = sum(c.values())
        doubled = c.get("postpositive_doubled", 0) + c.get("prepositive_doubled", 0)
        print(f"  {taam:<18}{tot:>7,} words · doubled {doubled:>6,} ({100 * doubled / tot:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="aligned dataset json with word_raw fields")
    args = ap.parse_args()
    self_test()
    if args.corpus:
        corpus_stats(args.corpus)


if __name__ == "__main__":
    main()
