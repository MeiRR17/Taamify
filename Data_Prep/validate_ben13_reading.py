"""
Validate Ben13's reading corpus against Sefaria, before trusting a word of it.
=============================================================================
The lesson from the ta'am-name clips is that Ben13's own text can disagree
with Ben13's own audio, and that a forced alignment given a wrong word list
fails silently rather than loudly. So this corpus gets checked before it is
used, not after.

Four independent checks per verse:

1. TEXT INTEGRITY — concatenating a verse's segments must reproduce Sefaria's
   verse exactly, letter for letter and accent for accent. This catches a
   missing segment, a duplicated one, a dropped word, or a different textual
   tradition.

2. TA'AM SEQUENCE — the ta'amim decoded from the concatenated segments must
   match the ta'amim decoded from Sefaria's verse, in order. Text can match
   while accents differ (different pointing), so this is a separate check.

3. CUT-POINT LEGITIMACY — every internal segment boundary should land on a
   DISJUNCTIVE accent. A cut in the middle of a conjunctive run would be
   musically wrong, and is the exact failure mode that ruined the earlier
   CTC-derived segments. Uses taam_grammar.py's own classification, which was
   derived independently of Ben13.

4. AUDIO PRESENT — the mp3 exists, is non-empty, and is long enough to hold
   the text it claims.

Nothing here is auto-corrected. Mismatches are reported so a human decides,
because the last time this project inferred a fix from a scoring number it
manufactured confident-looking bad data.

Usage:
  python Data_Prep/validate_ben13_reading.py
  python Data_Prep/validate_ben13_reading.py --verbose
"""
import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taam_grammar as tg  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "data" / "ben13_reading" / "manifest_yerushalmi.json"
CACHE = PROJECT_DIR / "Data_Prep" / "sefaria_cache"

BOOK_NAMES = {"1": "Genesis", "2": "Exodus", "3": "Leviticus",
              "4": "Numbers", "5": "Deuteronomy"}

# characters that carry no phonetic or accentual content for this comparison
STRIP = {"־",   # maqaf (hyphen) -- Sefaria and Ben13 differ on spacing
         "׃",   # sof pasuq
         "׀",   # paseq
         "‎", "‏", " "}


# Sefaria's stored text carries presentation markup -- <big>, <small>,
# &thinsp;, and parasha markers like <span class="mam-spi-pe">{פ}</span>.
# The first validation run counted every one of those as a text mismatch,
# which is why it reported 26% agreement when the real disagreement was far
# smaller. Strip them before comparing; they are typography, not text.
HTML_TAG = re.compile(r"<[^>]+>")


def norm(s: str) -> str:
    s = html.unescape(HTML_TAG.sub("", s or ""))
    s = unicodedata.normalize("NFC", s)
    s = "".join(" " if c in STRIP else c for c in s)
    return " ".join(s.split())


def sefaria_verse(book: str, perek: int, pasuk: int):
    p = CACHE / f"{book}.{perek}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        verses = json.load(f)
    return verses[pasuk - 1] if 0 < pasuk <= len(verses) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        rows = json.load(f)

    by_verse = {}
    for r in rows:
        key = (r["book"], int(r["perek"]), int(r["pasuk"]))
        by_verse.setdefault(key, []).append(r)
    for v in by_verse.values():
        v.sort(key=lambda r: int(r["block"]))

    stats = {"verses": 0, "text_ok": 0, "taam_ok": 0, "cuts_ok": 0,
             "audio_ok": 0, "segments": 0}
    problems = []

    for (book_id, perek, pasuk), segs in sorted(by_verse.items()):
        book = BOOK_NAMES.get(book_id)
        raw = sefaria_verse(book, perek, pasuk)
        if raw is None:
            problems.append((f"{book} {perek}:{pasuk}", "no Sefaria text cached"))
            continue
        stats["verses"] += 1
        stats["segments"] += len(segs)
        ref = f"{book} {perek}:{pasuk}"

        # --- 1. text integrity ---
        joined = norm(" ".join(s["text"] for s in segs))
        expect = norm(raw)
        if joined == expect:
            stats["text_ok"] += 1
        else:
            problems.append((ref, "TEXT differs from Sefaria"))
            if args.verbose:
                print(f"  {ref}\n    ben13:   {joined[:90]}\n    sefaria: {expect[:90]}")

        # --- 2. ta'am sequence ---
        b_taams = [t for _, t in tg.verse_tokens(" ".join(s["text"] for s in segs)) if t]
        s_taams = [t for _, t in tg.verse_tokens(raw) if t]
        if b_taams == s_taams:
            stats["taam_ok"] += 1
        else:
            problems.append((ref, f"TA'AMIM differ ({len(b_taams)} vs {len(s_taams)})"))
            if args.verbose:
                print(f"  {ref}\n    ben13:   {b_taams}\n    sefaria: {s_taams}")

        # --- 3. cut points land on disjunctives ---
        bad_cuts = []
        for s in segs[:-1]:                       # last segment ends the verse
            toks = tg.verse_tokens(s["text"])
            last = toks[-1][1] if toks else None
            if last not in tg.DISJUNCTIVE_RANK:
                bad_cuts.append((s["block"], last))
        if not bad_cuts:
            stats["cuts_ok"] += 1
        else:
            problems.append((ref, f"CUT on non-disjunctive: {bad_cuts}"))

        # --- 4. audio present ---
        missing = [s["block"] for s in segs
                   if not (PROJECT_DIR / s["local_path"]).exists()
                   or (PROJECT_DIR / s["local_path"]).stat().st_size < 1000]
        if not missing:
            stats["audio_ok"] += 1
        else:
            problems.append((ref, f"AUDIO missing/tiny for blocks {missing}"))

    n = max(stats["verses"], 1)
    print(f"\n{'=' * 66}")
    print(f"  verses checked : {stats['verses']}   segments: {stats['segments']}")
    print(f"  text matches Sefaria      : {stats['text_ok']:>4} / {n}  "
          f"({100*stats['text_ok']/n:.1f}%)")
    print(f"  ta'am sequence matches    : {stats['taam_ok']:>4} / {n}  "
          f"({100*stats['taam_ok']/n:.1f}%)")
    print(f"  all cuts on disjunctives  : {stats['cuts_ok']:>4} / {n}  "
          f"({100*stats['cuts_ok']/n:.1f}%)")
    print(f"  audio present             : {stats['audio_ok']:>4} / {n}  "
          f"({100*stats['audio_ok']/n:.1f}%)")
    print(f"{'=' * 66}")

    if problems:
        print(f"\n{len(problems)} issue(s):")
        for ref, msg in problems[:40]:
            print(f"  {ref:<18} {msg}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
    else:
        print("\nno issues found.")


if __name__ == "__main__":
    main()
