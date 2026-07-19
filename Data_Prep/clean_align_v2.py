"""
Precision-First Torah Cleaning & Alignment Pipeline (v2)
========================================================
Replaces torah_cleaning_pipeline.py. One positional alignment that does everything:

  whisper JSON (word timestamps)  x  Sefaria canonical text (with ta'amim)
      -> verified words: {word, taam, start, end, verse_ref, probability}

Fixes over v1:
- Verse-exact parasha boundaries (v1 mapped whole chapters, wrong for nearly
  all 54 parashiyot -> massive silent data loss).
- Positional 'equal'-only alignment (v1 whitelisted by word *set*, readmitting
  every unverified duplicate occurrence).
- Ta'am is taken from the SAME alignment (v1 attached ta'amim in a second,
  separate alignment pass).
- Complete ta'am unicode map (v1 lost Revia, Segol, Zaqef_Gadol, Mercha_Kefula
  and mislabeled Zarqa/Zinor).
- Sefaria responses cached locally (Data_Prep/sefaria_cache/).

Precision-first: only exact word matches inside runs of >= MIN_RUN consecutive
matches survive. No Levenshtein "corrections" by default.

Usage:
  python Data_Prep/clean_align_v2.py                       # all readers
  python Data_Prep/clean_align_v2.py --readers R1 R2       # subset
  python Data_Prep/clean_align_v2.py --min-prob 0.3        # stricter
"""

import argparse
import html
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_DIR / "metadata"
CLEANED_DIR = PROJECT_DIR / "metadata_cleaned_v2"
CACHE_DIR = Path(__file__).resolve().parent / "sefaria_cache"
OLD_CLEANED_DIR = PROJECT_DIR / "metadata_cleaned"

SEFARIA_API = "https://www.sefaria.org/api/texts/{ref}?context=0&commentary=0"

CHAPTERS_PER_BOOK = {
    "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36, "Deuteronomy": 34,
}

# Verse-exact parasha boundaries: (book, start_chapter, start_verse, end_chapter, end_verse)
# Standard Hebrew (Masoretic) versification, as used by Sefaria.
PARASHA_BOUNDS = {
    "01_bereshit":        ("Genesis", 1, 1, 6, 8),
    "02_noah":            ("Genesis", 6, 9, 11, 32),
    "03_lech_lecha":      ("Genesis", 12, 1, 17, 27),
    "04_vayera":          ("Genesis", 18, 1, 22, 24),
    "05_chayei_sarah":    ("Genesis", 23, 1, 25, 18),
    "06_toledot":         ("Genesis", 25, 19, 28, 9),
    "07_vayetze":         ("Genesis", 28, 10, 32, 3),
    "08_vayishlach":      ("Genesis", 32, 4, 36, 43),
    "09_vayeshev":        ("Genesis", 37, 1, 40, 23),
    "10_miketz":          ("Genesis", 41, 1, 44, 17),
    "11_vayigash":        ("Genesis", 44, 18, 47, 27),
    "12_vayechi":         ("Genesis", 47, 28, 50, 26),
    "13_shemot":          ("Exodus", 1, 1, 6, 1),
    "14_vaera":           ("Exodus", 6, 2, 9, 35),
    "15_bo":              ("Exodus", 10, 1, 13, 16),
    "16_beshalach":       ("Exodus", 13, 17, 17, 16),
    "17_yitro":           ("Exodus", 18, 1, 20, 23),
    "18_mishpatim":       ("Exodus", 21, 1, 24, 18),
    "19_terumah":         ("Exodus", 25, 1, 27, 19),
    "20_tetzaveh":        ("Exodus", 27, 20, 30, 10),
    "21_ki_tisa":         ("Exodus", 30, 11, 34, 35),
    "22_vayakhel":        ("Exodus", 35, 1, 38, 20),
    "23_pekudei":         ("Exodus", 38, 21, 40, 38),
    "24_vayikra":         ("Leviticus", 1, 1, 5, 26),
    "25_tzav":            ("Leviticus", 6, 1, 8, 36),
    "26_shemini":         ("Leviticus", 9, 1, 11, 47),
    "27_tazria":          ("Leviticus", 12, 1, 13, 59),
    "28_metzora":         ("Leviticus", 14, 1, 15, 33),
    "29_acharei_mot":     ("Leviticus", 16, 1, 18, 30),
    "30_kedoshim":        ("Leviticus", 19, 1, 20, 27),
    "31_emor":            ("Leviticus", 21, 1, 24, 23),
    "32_behar":           ("Leviticus", 25, 1, 26, 2),
    "33_bechukotai":      ("Leviticus", 26, 3, 27, 34),
    "34_bamidbar":        ("Numbers", 1, 1, 4, 20),
    "35_naso":            ("Numbers", 4, 21, 7, 89),
    "36_behaalotecha":    ("Numbers", 8, 1, 12, 16),
    "37_shlach":          ("Numbers", 13, 1, 15, 41),
    "38_korach":          ("Numbers", 16, 1, 18, 32),
    "39_chukat":          ("Numbers", 19, 1, 22, 1),
    "40_balak":           ("Numbers", 22, 2, 25, 9),
    "41_pinchas":         ("Numbers", 25, 10, 30, 1),
    "42_matot":           ("Numbers", 30, 2, 32, 42),
    "43_masei":           ("Numbers", 33, 1, 36, 13),
    "44_devarim":         ("Deuteronomy", 1, 1, 3, 22),
    "45_vaetchanan":      ("Deuteronomy", 3, 23, 7, 11),
    "46_eikev":           ("Deuteronomy", 7, 12, 11, 25),
    "47_reeh":            ("Deuteronomy", 11, 26, 16, 17),
    "48_shoftim":         ("Deuteronomy", 16, 18, 21, 9),
    "49_ki_teitzei":      ("Deuteronomy", 21, 10, 25, 19),
    "50_ki_tavo":         ("Deuteronomy", 26, 1, 29, 8),
    "51_nitzavim":        ("Deuteronomy", 29, 9, 30, 20),
    "52_vayelech":        ("Deuteronomy", 31, 1, 31, 30),
    "53_haazinu":         ("Deuteronomy", 32, 1, 32, 52),
    "54_vezot_haberacha": ("Deuteronomy", 33, 1, 34, 12),
}

# Complete ta'am unicode map (U+0591-U+05AE). Names match the existing class
# naming convention (classes_top5.json / Data_Prep/dataset_full).
# Both U+0598 (poetic tsinnorit glyph) and U+05AE (prose zarqa) map to Zarqa:
# in Torah prose the zarqa mark is encoded U+05AE; v1's "Zinor" class was an
# artifact of this encoding split.
TAAM_MAP = {
    "֑": "Etnachta",
    "֒": "Segol",
    "֓": "Shalshelet",
    "֔": "Zaqef_Qatan",
    "֕": "Zaqef_Gadol",
    "֖": "Tipecha",
    "֗": "Revia",
    "֘": "Zarqa",
    "֙": "Pashta",
    "֚": "Yetiv",
    "֛": "Tevir",
    "֜": "Geresh",
    "֝": "Geresh_Muqdam",
    "֞": "Gershayim",
    "֟": "Qarney_Para",
    "֠": "Telisha_Gedola",
    "֡": "Pazer",
    "֣": "Munach",
    "֤": "Mahapakh",
    "֥": "Mercha",
    "֦": "Mercha_Kefula",
    "֧": "Darga",
    "֨": "Qadma",
    "֩": "Telisha_Qetana",
    "֪": "Yerach_Ben_Yomo",
    "֮": "Zarqa",
}
# Not ta'amim: U+05A2/05AB/05AC/05AD are poetic-book marks (absent from Torah),
# U+05AF masora circle, U+05BD meteg, U+05C0 paseq, U+05C3 sof pasuq.

MAQAF = "־"
HEBREW_LETTERS = re.compile(r"[א-ת]+")
NON_LETTERS = re.compile(r"[^א-ת]")
HTML_TAGS = re.compile(r"<[^>]+>")
CURLY_MARKERS = re.compile(r"\{[^}]*\}")  # {פ} {ס} {ר} section markers
VAV_YOD = re.compile(r"[וי]")


def normalize_word(token: str) -> str:
    """Reduce a token to bare Hebrew letters (no nikud/ta'amim/punctuation)."""
    token = unicodedata.normalize("NFC", token)
    return NON_LETTERS.sub("", token)


def skeleton(norm: str) -> str:
    """
    Matching key that bridges ktiv male (Whisper: אלוהים) and ktiv haser
    (Torah: אלהים): drop vav/yod. Used ONLY for alignment; the canonical
    Sefaria form is what gets stored.
    """
    return VAV_YOD.sub("", norm)


def extract_taamim(token: str) -> list:
    """Ordered list of ta'am names appearing in a raw (cantillated) token."""
    token = unicodedata.normalize("NFC", token)
    return [TAAM_MAP[ch] for ch in token if ch in TAAM_MAP]


def fetch_chapter(book: str, chapter: int, session: requests.Session) -> list:
    """Fetch one chapter (list of cantillated verse strings), with local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{book}.{chapter}.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    url = SEFARIA_API.format(ref=f"{book}.{chapter}")
    for attempt in range(5):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            verses = resp.json().get("he", [])
            if isinstance(verses, str):
                verses = [verses]
            if not verses:
                raise ValueError(f"empty 'he' text for {book}.{chapter}")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(verses, f, ensure_ascii=False)
            time.sleep(0.3)  # be polite to the API
            return verses
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            print(f"    Sefaria fetch {book}.{chapter} failed ({e}), retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {book}.{chapter} from Sefaria")


def build_sefaria_words(parasha: str, session: requests.Session) -> list:
    """
    Ordered token list for a parasha (verse-exact boundaries).
    Each token: {norm, taamim, taam, verse_ref}. Maqaf-joined words are split;
    only the ta'am-bearing member of the chain carries a ta'am.
    """
    book, sc, sv, ec, ev = PARASHA_BOUNDS[parasha]
    words = []
    for ch in range(sc, ec + 1):
        verses = fetch_chapter(book, ch, session)
        v_from = sv if ch == sc else 1
        v_to = ev if ch == ec else len(verses)
        if ch == ec and ev > len(verses):
            raise ValueError(
                f"{parasha}: boundary {book} {ch}:{ev} beyond chapter end ({len(verses)} verses)"
            )
        for v in range(v_from, v_to + 1):
            raw_verse = verses[v - 1]
            raw_verse = html.unescape(raw_verse)
            # Tags wrap inline letters (e.g. the enlarged Bet of Genesis 1:1),
            # so they must vanish without splitting the word.
            raw_verse = HTML_TAGS.sub("", raw_verse)
            raw_verse = CURLY_MARKERS.sub(" ", raw_verse)
            raw_verse = raw_verse.replace(MAQAF, " ")
            verse_words = []
            for raw_token in raw_verse.split():
                norm = normalize_word(raw_token)
                if len(norm) < 2:  # no 1-letter words in Torah text
                    continue
                taamim = extract_taamim(raw_token)
                verse_words.append({
                    "norm": norm,
                    "taamim": taamim,
                    "taam": taamim[-1] if taamim else None,
                    "verse_ref": f"{book} {ch}:{v}",
                })
            # The last word of every verse is sung with Siluk (sof pasuq melody).
            # Unicode encodes silluq with the meteg codepoint, so TAAM_MAP can't
            # see it — assign it positionally.
            if verse_words and verse_words[-1]["taam"] is None:
                verse_words[-1]["taam"] = "Siluk"
                verse_words[-1]["taamim"] = verse_words[-1]["taamim"] + ["Siluk"]
            words.extend(verse_words)
    return words


def load_whisper_words(metadata_path: Path) -> list:
    """Flat word list from a whisper JSON: {norm, raw, start, end, probability}."""
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            raw = w.get("word", "").strip()
            norm = normalize_word(raw)
            if len(norm) < 2:
                continue
            words.append({
                "norm": norm,
                "raw": raw,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": float(w.get("probability", 0.0)),
            })
    return words


def align(whisper_words: list, sefaria_words: list, min_prob: float,
          min_run: int, min_dur: float, max_dur: float) -> tuple:
    """
    Positional equal-only alignment. Returns (verified_records, stats).

    Matching runs on the ktiv-male/haser-bridging skeleton (no vav/yod), so
    Whisper's modern spelling still matches the canonical text. Precision
    guards: a run must be >= min_run long AND contain at least one full exact
    match (or be longer than min_run) — a lone skeleton-only run is rejected.
    Each surviving word must also pass probability >= min_prob, duration in
    [min_dur, max_dur], and its Sefaria token must bear a ta'am.
    """
    import difflib

    stats = Counter()
    stats["whisper_words"] = len(whisper_words)
    stats["sefaria_words"] = len(sefaria_words)

    a = [skeleton(w["norm"]) for w in whisper_words]
    b = [skeleton(s["norm"]) for s in sefaria_words]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    records = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            stats["dropped_unmatched"] += i2 - i1
            continue
        run_len = i2 - i1
        if run_len < min_run:
            stats["dropped_short_run"] += run_len
            continue
        n_exact = sum(
            1 for k in range(run_len)
            if whisper_words[i1 + k]["norm"] == sefaria_words[j1 + k]["norm"]
        )
        if n_exact == 0 and run_len <= min_run:
            stats["dropped_no_exact_anchor"] += run_len
            continue
        for k in range(run_len):
            w = whisper_words[i1 + k]
            s = sefaria_words[j1 + k]
            stats["matched"] += 1
            if w["probability"] < min_prob:
                stats["dropped_low_prob"] += 1
                continue
            dur = w["end"] - w["start"]
            if not (min_dur <= dur <= max_dur):
                stats["dropped_bad_duration"] += 1
                continue
            if s["taam"] is None:
                stats["dropped_no_taam"] += 1  # e.g. non-final maqaf-chain member
                continue
            records.append({
                "word_clean": s["norm"],
                "taam": s["taam"],
                "taamim_all": s["taamim"],
                "verse_ref": s["verse_ref"],
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
                "probability": round(w["probability"], 4),
                "match": "exact" if w["norm"] == s["norm"] else "skeleton",
            })
            stats["verified"] += 1
    return records, stats


def discover_recordings(reader: str) -> list:
    """(metadata_path, parasha_key) pairs for a reader, e.g. r1_01_bereshit.json -> 01_bereshit."""
    reader_dir = METADATA_DIR / reader
    if not reader_dir.exists():
        return []
    out = []
    for p in sorted(reader_dir.glob("*.json")):
        stem = re.sub(r"^r\d+_", "", p.stem)
        if stem in PARASHA_BOUNDS:
            out.append((p, stem))
        else:
            print(f"  WARNING: {p.name}: no parasha for '{stem}', skipping")
    return out


def count_old_cleaned_words(reader: str) -> int:
    """Word count in the v1 'perfect_clean' output, for the audit comparison."""
    total = 0
    old_dir = OLD_CLEANED_DIR / reader
    if not old_dir.exists():
        return 0
    for p in old_dir.glob("*_perfect_clean.json"):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            total += sum(len(s.get("words", [])) for s in d.get("segments", []))
        except (json.JSONDecodeError, OSError):
            pass
    return total


def main():
    ap = argparse.ArgumentParser(description="Precision-first Torah cleaning v2")
    ap.add_argument("--readers", nargs="+", default=["R1", "R2", "R3", "R4", "R5"])
    ap.add_argument("--min-prob", type=float, default=0.2)
    ap.add_argument("--min-run", type=int, default=2,
                    help="minimum consecutive exact matches for a word to count")
    ap.add_argument("--min-dur", type=float, default=0.1)
    # Sung words stretch well past 2 s (ornate ta'amim like Pazer/Telisha are the
    # longest); the 87-frame window truncates the tail, but dropping them would
    # bias the dataset against exactly those classes.
    ap.add_argument("--max-dur", type=float, default=3.0)
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Taamify-research (student project)"

    all_records = []
    audit = {}          # reader -> parasha -> stats
    taam_counts = Counter()
    reader_taam_counts = defaultdict(Counter)

    for reader in args.readers:
        recordings = discover_recordings(reader)
        print(f"\n{'='*60}\n{reader}: {len(recordings)} recordings\n{'='*60}")
        audit[reader] = {}
        for meta_path, parasha in recordings:
            whisper_words = load_whisper_words(meta_path)
            if not whisper_words:
                print(f"  {parasha}: NO word timestamps in {meta_path.name} — needs re-transcription")
                audit[reader][parasha] = {"whisper_words": 0, "verified": 0,
                                          "sefaria_words": 0, "error": "no_word_timestamps"}
                continue
            sefaria_words = build_sefaria_words(parasha, session)
            records, stats = align(whisper_words, sefaria_words,
                                   args.min_prob, args.min_run,
                                   args.min_dur, args.max_dur)

            audio_file = f"data/{reader}/wav/{parasha}.wav"
            for r in records:
                r_out = {"reader": reader, "parasha": parasha, "audio_file": audio_file, **r}
                all_records.append(r_out)
                taam_counts[r["taam"]] += 1
                reader_taam_counts[reader][r["taam"]] += 1

            out_dir = CLEANED_DIR / reader
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{parasha}.json", "w", encoding="utf-8") as f:
                json.dump({"reader": reader, "parasha": parasha,
                           "audio_file": audio_file, "stats": dict(stats),
                           "words": records}, f, ensure_ascii=False, indent=1)

            audit[reader][parasha] = dict(stats)
            pct = 100.0 * stats["verified"] / max(stats["whisper_words"], 1)
            cov = 100.0 * stats["verified"] / max(stats["sefaria_words"], 1)
            print(f"  {parasha}: {stats['whisper_words']} whisper -> "
                  f"{stats['verified']} verified ({pct:.0f}% kept, {cov:.0f}% of canon)")

    # Combined dataset — merge with previous runs (a run covers only its
    # --readers; records of other readers are preserved)
    combined_path = PROJECT_DIR / "aligned_dataset_v2.json"
    if combined_path.exists():
        with open(combined_path, encoding="utf-8") as f:
            previous = [r for r in json.load(f) if r["reader"] not in args.readers]
        if previous:
            print(f"Keeping {len(previous)} records of other readers from previous runs")
            all_records = previous + all_records
            for r in previous:
                taam_counts[r["taam"]] += 1
                reader_taam_counts[r["reader"]][r["taam"]] += 1
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    print(f"\nSaved {len(all_records)} verified words -> {combined_path.name}")

    # Audit report
    report = ["# Data Audit Report — Cleaning v2 (precision-first)", "",
              f"Parameters: min_prob={args.min_prob}, min_run={args.min_run}, "
              f"duration=[{args.min_dur},{args.max_dur}]s", ""]
    report.append("## Per-reader summary (all readers cleaned so far)\n")
    report.append("| Reader | Recordings | Whisper words | Verified | Kept % | v1 cleaned (old) |")
    report.append("|--------|-----------|---------------|----------|--------|------------------|")
    all_reader_dirs = sorted(d.name for d in CLEANED_DIR.iterdir() if d.is_dir())
    for reader in all_reader_dirs:
        ww = vv = n_rec = 0
        for pjson in (CLEANED_DIR / reader).glob("*.json"):
            with open(pjson, encoding="utf-8") as f:
                s = json.load(f).get("stats", {})
            ww += s.get("whisper_words", 0)
            vv += s.get("verified", 0)
            n_rec += 1
        old = count_old_cleaned_words(reader)
        pct = 100.0 * vv / max(ww, 1)
        report.append(f"| {reader} | {n_rec} | {ww} | {vv} | {pct:.1f}% | {old} |")

    report.append("\n## Ta'am distribution (all readers)\n")
    report.append("| # | Ta'am | Count |")
    report.append("|---|-------|-------|")
    for i, (taam, n) in enumerate(taam_counts.most_common(), 1):
        report.append(f"| {i} | {taam} | {n} |")

    report.append("\n## Ta'am distribution per reader\n")
    readers_present = [r for r in args.readers if reader_taam_counts.get(r)]
    header = "| Ta'am | " + " | ".join(readers_present) + " |"
    report.append(header)
    report.append("|" + "---|" * (len(readers_present) + 1))
    for taam, _ in taam_counts.most_common():
        row = [str(reader_taam_counts[r].get(taam, 0)) for r in readers_present]
        report.append(f"| {taam} | " + " | ".join(row) + " |")

    report.append("\n## Per-parasha detail\n")
    for reader in args.readers:
        rec = audit.get(reader, {})
        if not rec:
            continue
        report.append(f"\n### {reader}\n")
        report.append("| Parasha | Whisper | Matched | Verified | Dropped: unmatched / short-run / low-prob / duration / no-taam |")
        report.append("|---------|---------|---------|----------|------|")
        for parasha, s in rec.items():
            if s.get("error"):
                report.append(f"| {parasha} | — | — | — | ERROR: {s['error']} |")
                continue
            drops = (f"{s.get('dropped_unmatched', 0)} / {s.get('dropped_short_run', 0)} / "
                     f"{s.get('dropped_low_prob', 0)} / {s.get('dropped_bad_duration', 0)} / "
                     f"{s.get('dropped_no_taam', 0)}")
            report.append(f"| {parasha} | {s.get('whisper_words', 0)} | {s.get('matched', 0)} | "
                          f"{s.get('verified', 0)} | {drops} |")

    report_path = PROJECT_DIR / "data_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"Audit report -> {report_path.name}")

    print("\nTa'am distribution:")
    for taam, n in taam_counts.most_common():
        print(f"  {taam:18s} {n}")


if __name__ == "__main__":
    main()
