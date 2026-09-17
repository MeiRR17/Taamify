"""
Sheva na or sheva nach, from the pointed text alone.
=====================================================
Station 6 was budgeted at 3-4 weeks because it was lumped together with
qamats qatan, which genuinely needs a morphological analyser. Sheva does
not: whether a sheva is pronounced (na) or silent (nach) follows classical
rules over the letters and vowels already present in the text.

The rules, in the order they are applied:

  1. word-initial sheva                    -> na
  2. sheva under a letter with dagesh
     chazak (a doubled letter)             -> na
  3. the SECOND of two consecutive shevas  -> na   (the first is nach)
  4. word-final sheva                      -> nach (final kaf keeps one,
                                             and it is silent)
  5. sheva after a LONG vowel              -> na
  6. sheva after a SHORT vowel             -> nach
  7. otherwise                             -> nach (the majority)

Rule 2 needs care: the dagesh codepoint is shared between dagesh qal, dagesh
chazak and mapiq. In begadkefat letters at the start of a syllable it is
usually qal, which does not force a na. The distinction that matters here is
whether the letter is DOUBLED, which shows up as a dagesh in a letter that
already closed the previous syllable -- i.e. one preceded by a vowel.

Why this matters acoustically: a sheva na is a real (very short) vowel and a
sheva nach is nothing at all. The difference is directly audible, and it is
one of the most common reading errors -- so it is a genuine feedback channel
rather than a spelling curiosity.

Self-test:
  python Data_Prep/sheva.py
"""
import unicodedata

SHEVA = "ְ"
DAGESH = "ּ"
# long vowels: qamats, tsere, holam, shuruq/holam-male are handled via maters
LONG = {"ָ", "ֵ", "ֹ", "ֻ"}       # qamats, tsere, holam, qubuts
SHORT = {"ַ", "ֶ", "ִ", "ׇ"}      # patah, segol, hiriq, qamats qatan
HATAF = {"ֱ", "ֲ", "ֳ"}                # hataf segol/patah/qamats
VOWELS = LONG | SHORT | HATAF
LETTERS = ("א", "ת")
FINAL_LETTERS = set("ךםןףץ")
MATER = set("אהוי")


def parse(word: str):
    """[(letter, [marks])] over the pointed word, cantillation stripped."""
    word = unicodedata.normalize("NFC", word)
    units = []
    for ch in word:
        if LETTERS[0] <= ch <= LETTERS[1]:
            units.append([ch, []])
        elif units and ("ְ" <= ch <= "ּ" or ch in ("ׇ", "ׁ", "ׂ")):
            units[-1][1].append(ch)
    return units


def vowel_of(marks):
    for m in marks:
        if m in VOWELS:
            return m
    return None


def classify(word: str):
    """[(letter_index, 'na'|'nach', rule)] for every sheva in the word."""
    units = parse(word)
    out = []
    sheva_idx = [i for i, (_, m) in enumerate(units) if SHEVA in m]
    for i in sheva_idx:
        letter, marks = units[i]

        if i == 0:
            out.append((i, "na", "word_initial"))
            continue
        if i == len(units) - 1:
            # a final sheva is silent; final kaf carries one by convention
            out.append((i, "nach", "word_final"))
            continue
        # doubled letter: dagesh in a letter that is itself preceded by a vowel
        prev_vowel = vowel_of(units[i - 1][1])
        if DAGESH in marks and prev_vowel is not None and SHEVA not in units[i - 1][1]:
            out.append((i, "na", "dagesh_chazak"))
            continue
        if SHEVA in units[i - 1][1]:
            # second of a pair -- the first was nach, this one is pronounced
            out.append((i, "na", "second_of_pair"))
            continue
        if prev_vowel == "\u05b8":
            # Qamats: the rule splits on something the pointing does not say.
            # After qamats GADOL (long /a/) the sheva is na; after qamats
            # QATAN (short /o/) it is nach. Telling them apart needs
            # morphology, not vowels -- קָטְלָה is qatan and its sheva is
            # silent, while שָׁמְרָה patterns the other way. Guessing here
            # would produce a confident wrong answer, so it is marked
            # ambiguous and no verdict is shown to the learner.
            out.append((i, "ambiguous", "after_qamats_needs_morphology"))
            continue
        if prev_vowel in LONG:
            out.append((i, "na", "after_long_vowel"))
            continue
        if prev_vowel in HATAF:
            out.append((i, "na", "after_hataf"))
            continue
        if prev_vowel in SHORT:
            out.append((i, "nach", "after_short_vowel"))
            continue
        out.append((i, "nach", "default"))
    return out


def word_label(word: str):
    """One label per word for the whole-word acoustic task: 'na' when the word
    contains at least one pronounced sheva, 'nach' when it has shevas but all
    are silent, None when it has none (nothing to hear either way)."""
    r = classify(word)
    if not r:
        return None
    kinds = [k for _, k, _ in r]
    if any(k == "ambiguous" for k in kinds):
        return None          # never label a word the rules cannot settle
    return "na" if "na" in kinds else "nach"


CASES = [
    # word,            expected kinds in order
    ("בְּרֵאשִׁ֖ית",     ["na"]),            # word-initial
    ("יִשְׁמְעוּ",        ["nach", "na"]),    # pair: first silent, second voiced
    ("מַלְכֵּי",          ["nach"]),          # after short patah
    ("שֹׁמְרִים",         ["na"]),            # after long holam
    ("וַיְדַבֵּ֥ר",       ["nach"]),          # vay-dabber: the sheva IS silent
    ("קָטְלָה",          ["ambiguous"]),     # qamats qatan -- rules alone cannot say
]


def self_test():
    ok = 0
    for word, expected in CASES:
        got = [k for _, k, _ in classify(word)]
        mark = "ok " if got == expected else "FAIL"
        if got == expected:
            ok += 1
        rules = [r for _, _, r in classify(word)]
        print(f"  {mark} {word:<14} got={got}  expected={expected}  rules={rules}")
    print(f"\n  {ok}/{len(CASES)} cases pass")
    return ok == len(CASES)


if __name__ == "__main__":
    self_test()
