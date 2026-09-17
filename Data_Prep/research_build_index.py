"""
Build a MATCHED cross-reader word index.
========================================
The research question is "what is invariant across readers", and the single
biggest threat to any answer is confounding: a ta'am correlates with word
length, with stress position, and with position in the verse. If we compare
Ben13's Tipecha words against R1's Tipecha words we are also comparing two
different bags of text.

This index removes the text confound by construction: it keys every word by
(book, chapter, verse, word-position-in-verse) and keeps only the keys that
at least two readers actually sang. Every cross-reader comparison downstream
is therefore the SAME WORD of the SAME VERSE, sung by different people.

Sources (deliberately two, because they disagree about what exists):
  * features_v3_word/BEN13R_meta.json -- Ben13, segment-level CTC. Audio is
    resolved through data/ben13_reading/manifest_yerushalmi.json by
    (parasha, perek, pasuk, block); start/end are offsets INSIDE that segment
    mp3, not inside a parasha-length wav.
  * aligned_dataset_merged.json -- R1/R2/R8, verse-level CTC into a single
    parasha wav (audio_file + absolute start/end). This file contains NO
    BEN13R rows, which is worth stating because the standing extraction job
    (Data_Prep/_melody4.py) filters it for "BEN13R" and will therefore
    silently produce a three-reader file.

Output: results/melody_research/matched_index.json
"""
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "Data_Prep"))
import stressed_syllable as ss  # noqa: E402

OUT = PROJECT_DIR / "results" / "melody_research" / "matched_index.json"
BOOKS = {"1": "Genesis", "2": "Exodus", "3": "Leviticus",
         "4": "Numbers", "5": "Deuteronomy"}


def ben13_audio_map():
    """(parasha, perek, pasuk, block) -> local mp3 path."""
    rows = json.loads((PROJECT_DIR / "data" / "ben13_reading" /
                       "manifest_yerushalmi.json").read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        key = (str(r["parasha"]), str(r["perek"]), str(r["pasuk"]), str(r["block"]))
        out[key] = r["local_path"]
    return out


def stress_info(word_raw):
    if not word_raw:
        return None
    try:
        s = ss.stressed_syllable(unicodedata.normalize("NFC", word_raw))
    except Exception:
        return None
    if not s.get("ok"):
        return None
    return {"n_syl": s["n_syllables"], "stress_idx": s["stress_index"],
            "from_end": s["from_end"], "n_letters": s["n_letters"],
            "letter_start": s["letter_start"], "letter_end": s["letter_end"]}


def load_ben13():
    meta = json.loads((PROJECT_DIR / "features_v3_word" /
                       "BEN13R_meta.json").read_text(encoding="utf-8"))
    amap = ben13_audio_map()
    # word order inside a verse: Ben13 is cut into blocks, so order by
    # (block number, start-time inside that block)
    rows = []
    for m in meta:
        parts = m["verse_ref"].split(".")
        if len(parts) != 3:
            continue
        book, perek, pasuk = parts
        key = (str(m["parasha"]), str(perek), str(pasuk), str(m.get("segment_block")))
        path = amap.get(key)
        if path is None:
            continue
        rows.append({
            "reader": "BEN13R", "taam": m["taam"], "word_clean": m["word_clean"],
            "word_raw": m.get("word_raw"),
            "ref": f"{BOOKS.get(book, book)} {perek}:{pasuk}",
            "audio_file": path, "start": m["start"], "end": m["end"],
            "ctc_score": m.get("ctc_score"),
            "low_confidence": bool(m.get("low_confidence")),
            "_order": (int(m.get("segment_block") or 0), m["start"]),
        })
    return rows


def load_merged():
    data = json.loads((PROJECT_DIR / "aligned_dataset_merged.json")
                      .read_text(encoding="utf-8"))
    rows = []
    for r in data:
        if r["reader"] not in ("R1", "R2", "R8"):
            continue
        rows.append({
            "reader": r["reader"], "taam": r["taam"],
            "word_clean": r["word_clean"], "word_raw": r.get("word_raw"),
            "ref": r["verse_ref"], "audio_file": r["audio_file"],
            "start": r["start"], "end": r["end"],
            "ctc_score": None, "low_confidence": False,
            "prob": r.get("probability"),
            "_order": (0, r["start"]),
        })
    return rows


def main():
    rows = load_ben13() + load_merged()
    print(f"{len(rows):,} rows  " +
          str(Counter(r["reader"] for r in rows).most_common()))

    # word position inside each (reader, verse)
    by_verse = defaultdict(list)
    for r in rows:
        by_verse[(r["reader"], r["ref"])].append(r)
    for rs in by_verse.values():
        rs.sort(key=lambda r: r["_order"])
        for i, r in enumerate(rs):
            r["widx"] = i
            r["n_words_in_verse"] = len(rs)

    # matched key: same verse, same word position, and the same consonantal
    # word text (a cheap guard against an alignment having dropped a word and
    # shifted every index after it)
    keyed = defaultdict(dict)
    for r in rows:
        keyed[(r["ref"], r["widx"], r["word_clean"])][r["reader"]] = r

    matched = {k: v for k, v in keyed.items() if len(v) >= 2}
    print(f"{len(keyed):,} distinct word slots, {len(matched):,} sung by >=2 readers")
    print("  coverage by reader-count: " +
          str(Counter(len(v) for v in keyed.values()).most_common()))

    # ta'am agreement: if two readers' alignments disagree about the ta'am on
    # the same word slot, one of them is misaligned -- drop it, loudly
    agree, disagree = 0, 0
    clean = []
    for (ref, widx, wc), per_reader in matched.items():
        taams = {r["taam"] for r in per_reader.values()}
        if len(taams) != 1:
            disagree += 1
            continue
        agree += 1
        taam = taams.pop()
        raws = [r["word_raw"] for r in per_reader.values() if r.get("word_raw")]
        st = stress_info(raws[0]) if raws else None
        clean.append({
            "ref": ref, "widx": widx, "word_clean": wc, "taam": taam,
            "word_raw": raws[0] if raws else None, "stress": st,
            "n_words_in_verse": max(r["n_words_in_verse"] for r in per_reader.values()),
            "readers": {rd: {"audio_file": r["audio_file"], "start": r["start"],
                             "end": r["end"], "ctc_score": r.get("ctc_score"),
                             "low_confidence": r.get("low_confidence", False),
                             "prob": r.get("prob")}
                        for rd, r in per_reader.items()},
        })
    print(f"  ta'am agrees on {agree:,}, disagrees on {disagree:,} "
          f"({disagree / max(1, agree + disagree):.1%} dropped)")

    n4 = sum(1 for c in clean if len(c["readers"]) == 4)
    print(f"  {n4:,} slots have all four readers")
    cls = Counter(c["taam"] for c in clean)
    print("\n  per ta'am (>=2 readers):")
    for t, n in cls.most_common():
        n4t = sum(1 for c in clean if c["taam"] == t and len(c["readers"]) == 4)
        print(f"    {t:<18}{n:>7,}   all4={n4t:>6,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(PROJECT_DIR)}  ({len(clean):,} slots)")


if __name__ == "__main__":
    main()
