"""
Attach the vocalized+cantillated spelling to records that only carry consonants.
================================================================================
clean_align_v2 stored `word_clean` — bare letters, with the nikud and the ta'am
marks stripped. That was enough to train on whole words, but the stressed
syllable is located *by the position of the ta'am inside the pointed spelling*,
so R1/R2 cannot be windowed without recovering it.

Every record carries its `verse_ref`, and the canonical verse is already in the
Sefaria cache. Within one verse the records are in reading order, so the pointed
tokens are recovered by aligning the two sequences positionally (difflib on the
consonantal skeleton) rather than by looking words up — a lookup would silently
mis-assign every repeated word in a verse, and repetition is common.

Usage:
  python Data_Prep/backfill_word_raw.py --dry-run
  python Data_Prep/backfill_word_raw.py --dataset aligned_dataset_merged.json
"""

import argparse
import difflib
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import (  # noqa: E402
    CACHE_DIR,
    CURLY_MARKERS,
    HTML_TAGS,
    MAQAF,
    extract_taamim,
    normalize_word,
    skeleton,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def verse_tokens(book: str, chapter: int, verse: int) -> list:
    """Pointed tokens of one verse, tokenized exactly as clean_align_v2 did."""
    cache_file = CACHE_DIR / f"{book}.{chapter}.json"
    if not cache_file.exists():
        return []
    with open(cache_file, encoding="utf-8") as f:
        verses = json.load(f)
    if verse > len(verses):
        return []
    raw = html.unescape(verses[verse - 1])
    raw = HTML_TAGS.sub("", raw)
    raw = CURLY_MARKERS.sub(" ", raw)
    raw = raw.replace(MAQAF, " ")
    out = []
    for tok in raw.split():
        norm = normalize_word(tok)
        if len(norm) < 2:
            continue
        out.append({"raw": tok, "norm": norm})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aligned_dataset_merged.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = PROJECT_DIR / args.dataset
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    todo = [r for r in records if not r.get("word_raw")]
    print(f"records: {len(records):,} · missing word_raw: {len(todo):,}")

    # Group by reader+recording+verse; within a verse the records are ordered.
    groups = defaultdict(list)
    for r in todo:
        groups[(r["reader"], r["parasha"], r["verse_ref"])].append(r)

    stats = Counter()
    for (_reader, _parasha, ref), items in groups.items():
        try:
            book_part, cv = ref.rsplit(" ", 1)
            chapter, verse = (int(x) for x in cv.split(":"))
        except ValueError:
            stats["bad_ref"] += len(items)
            continue
        tokens = verse_tokens(book_part, chapter, verse)
        if not tokens:
            stats["no_verse"] += len(items)
            continue

        items.sort(key=lambda r: r["start"])
        a = [skeleton(r["word_clean"]) for r in items]
        b = [skeleton(t["norm"]) for t in tokens]
        matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                stats["unmatched"] += (i2 - i1)
                continue
            for k in range(i2 - i1):
                rec, tok = items[i1 + k], tokens[j1 + k]
                # Guard the alignment: the stored ta'am must be the one the
                # pointed token actually bears, or the row is not this word.
                taamim = extract_taamim(tok["raw"])
                expected = rec["taam"]
                if expected == "Siluk":
                    ok = not taamim or taamim[-1] == "Siluk"
                else:
                    ok = bool(taamim) and taamim[-1] == expected
                if not ok:
                    stats["taam_mismatch"] += 1
                    continue
                rec["word_raw"] = tok["raw"]
                stats["filled"] += 1

    still = sum(1 for r in records if not r.get("word_raw"))
    print(f"\nfilled          : {stats['filled']:,}")
    for k in ("unmatched", "taam_mismatch", "no_verse", "bad_ref"):
        if stats[k]:
            print(f"{k:<16}: {stats[k]:,}")
    print(f"still missing   : {still:,} "
          f"({100 * still / max(len(records), 1):.1f}% of the dataset)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\nupdated {path.name}")


if __name__ == "__main__":
    main()
