"""
The ta'amim as a grammar: verse division by continuous dichotomy.
=================================================================
The Masoretic accents are not decoration and not merely a melody index --
they are a punctuation system that recursively divides each verse into
two parts, each part into two again, down to the single word. Wu & Lowery
(LREC 2006, "A Hebrew Tree Bank Based on Cantillation Marks") coded this
as a formal grammar, parsed the whole Hebrew Bible with it, and found
that **over 99% of verses receive exactly one parse**.

That structure is what turns a drill app into a teaching one. A trainer
that only plays audio can say "wrong ta'am." A system that knows the tree
can say *why* the verse breaks where it does -- which is the actual
subject being taught.

THE TWO CLASSES
---------------
Every ta'am is either DISJUNCTIVE (מפסיק -- ends a phrase, may split a
span) or CONJUNCTIVE (משרת -- binds a word to the next, never splits).
The disjunctives are ranked by strength, traditionally by a royal
metaphor:

  rank 1  קיסרים   Silluq, Etnachta        -- divide the verse itself
  rank 2  מלכים    Zaqef (both), Tipcha,
                   Segol, Shalshelet       -- divide a half-verse
  rank 3  משנים    Revia, Zarqa, Pashta,
                   Yetiv, Tevir            -- divide a clause
  rank 4  שלישים   Geresh, Gershayim,
                   Pazer, Telisha Gedola,
                   Qarney Para             -- divide a phrase

PARSING RULE (continuous dichotomy)
-----------------------------------
Within any span, find the strongest disjunctive that is not the span's
own terminating accent, split immediately after it, and recurse on both
halves. A span with no internal disjunctive is a leaf.

SCOPE AND HONESTY
-----------------
This implements the 21-book prose system (Torah, Nevi'im, most Ketuvim).
The three poetic books -- Job, Proverbs, Psalms (אמ"ת) -- use a different
accent system entirely and are NOT handled here; they are detected and
refused rather than silently mis-parsed.

Usage:
  python Data_Prep/taam_grammar.py                    # parse + validate
  python Data_Prep/taam_grammar.py --verse "Genesis 1:1"
"""

import argparse
import glob
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import TAAM_MAP, CURLY_MARKERS, HTML_TAGS, MAQAF  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "Data_Prep" / "sefaria_cache"

# Disjunctives by rank -- lower rank binds the larger span.
DISJUNCTIVE_RANK = {
    "Siluk": 1, "Etnachta": 1,
    "Zaqef_Qatan": 2, "Zaqef_Gadol": 2, "Tipecha": 2,
    "Segol": 2, "Shalshelet": 2,
    "Revia": 3, "Zarqa": 3, "Pashta": 3, "Yetiv": 3, "Tevir": 3,
    "Geresh": 4, "Geresh_Muqdam": 4, "Gershayim": 4,
    "Pazer": 4, "Telisha_Gedola": 4, "Qarney_Para": 4,
}

# Conjunctives never divide anything -- they bind rightward to the next word.
CONJUNCTIVE = {
    "Munach", "Mercha", "Mahapakh", "Darga", "Qadma",
    "Telisha_Qetana", "Yerach_Ben_Yomo", "Mercha_Kefula",
}

# The Hebrew names a learner actually hears, per ta'am.
TAAM_HE = {
    "Siluk": "סילוק", "Etnachta": "אתנחתא", "Segol": "סגול",
    "Shalshelet": "שלשלת", "Zaqef_Qatan": "זקף קטן", "Zaqef_Gadol": "זקף גדול",
    "Tipecha": "טפחא", "Revia": "רביע", "Zarqa": "זרקא", "Pashta": "פשטא",
    "Yetiv": "יתיב", "Tevir": "תביר", "Geresh": "גרש",
    "Geresh_Muqdam": "גרש מוקדם", "Gershayim": "גרשיים", "Pazer": "פזר",
    "Telisha_Gedola": "תלישא גדולה", "Qarney_Para": "קרני פרה",
    "Munach": "מונח", "Mercha": "מרכא", "Mahapakh": "מהפך", "Darga": "דרגא",
    "Qadma": "קדמא", "Telisha_Qetana": "תלישא קטנה",
    "Yerach_Ben_Yomo": "ירח בן יומו", "Mercha_Kefula": "מרכא כפולה",
}

RANK_HE = {1: "קיסר", 2: "מלך", 3: "משנה", 4: "שליש"}

# Silluq shares its codepoint with meteg (U+05BD); it only counts as Silluq
# on the verse's final word, which is also the only place sof-pasuq appears.
SILLUQ_MARK = "ֽ"
SOF_PASUQ = "׃"

POETIC_BOOKS = {"Psalms", "Proverbs", "Job"}


def verse_tokens(raw: str) -> list:
    """[(word, taam_or_None)] for one pointed verse string."""
    txt = CURLY_MARKERS.sub(" ", HTML_TAGS.sub("", html.unescape(raw)))
    txt = txt.replace(MAQAF, " ").replace(SOF_PASUQ, " ")
    out = []
    words = [w for w in txt.split() if w.strip()]
    for i, w in enumerate(words):
        marks = [TAAM_MAP[c] for c in w if c in TAAM_MAP]
        taam = marks[-1] if marks else None
        # meteg on the last word is Silluq; elsewhere it's just a stress mark
        if taam is None and SILLUQ_MARK in w and i == len(words) - 1:
            taam = "Siluk"
        out.append((w, taam))
    return out


def parse_span(tokens, lo, hi):
    """Recursively divide tokens[lo:hi] at its strongest internal
    disjunctive. Returns a nested dict tree."""
    if hi - lo <= 1:
        w, t = tokens[lo]
        return {"type": "word", "word": w, "taam": t,
                "start": lo, "end": hi}

    # the span's own terminating accent doesn't divide it -- it closes it
    best_i, best_rank = None, 99
    for i in range(lo, hi - 1):
        t = tokens[i][1]
        r = DISJUNCTIVE_RANK.get(t)
        if r is not None and r < best_rank:
            best_rank, best_i = r, i

    if best_i is None:
        # no internal disjunctive: a chain of conjunctives running into the
        # span's own accent -- a leaf phrase, not divisible further
        return {"type": "phrase", "start": lo, "end": hi,
                "words": [w for w, _ in tokens[lo:hi]],
                "taam": tokens[hi - 1][1]}

    return {
        "type": "split",
        "start": lo, "end": hi,
        "divider": tokens[best_i][1],
        "divider_rank": best_rank,
        "divider_word": tokens[best_i][0],
        "children": [parse_span(tokens, lo, best_i + 1),
                     parse_span(tokens, best_i + 1, hi)],
    }


def parse_verse(raw: str):
    tokens = verse_tokens(raw)
    if not tokens:
        return None, []
    return parse_span(tokens, 0, len(tokens)), tokens


def describe(node, tokens, depth=0, out=None):
    """Human-readable Hebrew explanation of the division -- the teaching
    layer. This is the string an app shows to answer 'why does the verse
    break here?'"""
    if out is None:
        out = []
    pad = "  " * depth
    if node["type"] == "split":
        d = node["divider"]
        words = " ".join(w for w, _ in tokens[node["start"]:node["end"]])
        out.append(f"{pad}{TAAM_HE.get(d, d)} ({RANK_HE[node['divider_rank']]}) "
                   f"מחלק: {words}")
        for c in node["children"]:
            describe(c, tokens, depth + 1, out)
    elif node["type"] == "phrase":
        words = " ".join(node["words"])
        t = node["taam"]
        out.append(f"{pad}└ {words}  [{TAAM_HE.get(t, t)}]")
    else:
        t = node["taam"]
        out.append(f"{pad}└ {node['word']}  [{TAAM_HE.get(t, t)}]")
    return out


def tree_depth(node):
    if node["type"] != "split":
        return 1
    return 1 + max(tree_depth(c) for c in node["children"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verse", help='e.g. "Genesis 1:1"')
    ap.add_argument("--limit", type=int, default=0, help="chapters to scan")
    args = ap.parse_args()

    if args.verse:
        book, cv = args.verse.rsplit(" ", 1)
        ch, vs = (int(x) for x in cv.split(":"))
        with open(CACHE_DIR / f"{book}.{ch}.json", encoding="utf-8") as f:
            raw = json.load(f)[vs - 1]
        tree, tokens = parse_verse(raw)
        clean = " ".join(w for w, _ in tokens)
        print(f"\n{args.verse}\n{clean}\n")
        print("\n".join(describe(tree, tokens)))
        print(f"\ndepth {tree_depth(tree)}, {len(tokens)} words")
        return

    files = sorted(glob.glob(str(CACHE_DIR / "*.json")))
    if args.limit:
        files = files[:args.limit]
    n_verses = n_ok = n_poetic = 0
    depths, unknown = [], {}
    for path in files:
        book = Path(path).stem.rsplit(".", 1)[0]
        if book in POETIC_BOOKS:
            n_poetic += 1
            continue
        with open(path, encoding="utf-8") as f:
            verses = json.load(f)
        for raw in verses:
            if not isinstance(raw, str) or not raw.strip():
                continue
            n_verses += 1
            tree, tokens = parse_verse(raw)
            if tree is None:
                continue
            for _, t in tokens:
                if t and t not in DISJUNCTIVE_RANK and t not in CONJUNCTIVE:
                    unknown[t] = unknown.get(t, 0) + 1
            n_ok += 1
            depths.append(tree_depth(tree))

    print(f"parsed {n_ok:,} / {n_verses:,} verses "
          f"({100 * n_ok / max(n_verses, 1):.2f}%)")
    print(f"skipped {n_poetic} poetic-book chapter file(s) (different accent system)")
    if depths:
        depths.sort()
        print(f"tree depth: min {depths[0]}, median {depths[len(depths)//2]}, "
              f"max {depths[-1]}")
    if unknown:
        print("\nta'amim seen but not classified (check these):")
        for t, c in sorted(unknown.items(), key=lambda x: -x[1]):
            print(f"  {t:<20}{c:,}")


if __name__ == "__main__":
    main()
