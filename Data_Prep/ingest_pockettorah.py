"""
PocketTorah -> aligned_dataset records (reader R6)
=================================================
PocketTorah ships *human-verified word-level timestamps* (data/torah/labels/
<Parsha>-<aliyah>.txt: one comma-separated onset per word) next to its audio.
No Whisper, no forced alignment, no alignment error.

The timestamps were produced against PocketTorah's own text — the Westminster
Leningrad Codex bundled at data/torah/json/<Book>.json — so that is what we
tokenize. (Sefaria's text differs from WLC on a few dozen maqaf joins, which
knocks ~16% of the aliyot out of exact token correspondence.) WLC is fully
vocalized and cantillated, so the raw token is kept on every record: the ta'am
mark's position inside it is what locates the stressed syllable downstream.

Only the *Torah* readings are ingested. The `-H` files are haftarot: Prophets
cantillation is a different melodic system for the same marks, so mixing them
into a Torah-trope classifier would poison the labels.

Timestamps are onsets: word i spans ts[i] .. ts[i+1]; the last word of an
aliyah runs to the end of the audio (ffprobe).

Usage:
  python Data_Prep/ingest_pockettorah.py --dry-run     # validate only
  python Data_Prep/ingest_pockettorah.py               # validate + write
"""

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import TAAM_MAP, normalize_word  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
PT_DIR = PROJECT_DIR / "data" / "_sources" / "pockettorah" / "data"
AUDIO_DIR = PT_DIR / "audio"
LABELS_DIR = PT_DIR / "torah" / "labels"
WLC_DIR = PT_DIR / "torah" / "json"
ALIYAH_JSON = PT_DIR / "aliyah.json"
OUT_JSON = PROJECT_DIR / "aligned_dataset_pt.json"
CLEANED_DIR = PROJECT_DIR / "metadata_cleaned_v2"

READER = "R6"
BOOKS = ("Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy")
MORPH_SEP = "/"          # WLC morphology divider, not part of the spelling


def key(name: str) -> str:
    """Fold PocketTorah's three naming conventions onto one key.

    'Achrei Mot' / 'achrei mot' / 'AchreiMot' -> 'achreimot'
    'Beha’alotcha' / 'Behaalotcha'            -> 'behaalotcha'
    """
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).lower())


def index_dir(directory: Path, suffix: str) -> dict:
    """{(parsha_key, aliyah_num): path} for '<Name>-<n>' files, skipping haftarot."""
    out = {}
    for p in directory.glob(f"*{suffix}"):
        m = re.match(r"^(.*)-([0-9]+|[hHmM])$", p.stem)
        if not m:
            continue
        stem, num = m.groups()
        if num.lower() in ("h", "m"):
            continue
        out[(key(stem), int(num))] = p
    return out


def load_wlc(book: str) -> list:
    """chapters[ch][verse] -> list of raw cantillated tokens (1-indexed lookup helper)."""
    with open(WLC_DIR / f"{book}.json", encoding="utf-8-sig") as f:
        chapters = json.load(f)["Tanach"]["tanach"]["book"]["c"]
    out = []
    for ch in chapters:
        if not isinstance(ch, dict):       # each book ends with a null padding entry
            continue
        verses = ch.get("v") or []
        if isinstance(verses, dict):
            verses = [verses]
        rows = []
        for v in verses:
            words = v.get("w") if isinstance(v, dict) else None
            if isinstance(words, str):
                words = [words]
            rows.append([w.replace(MORPH_SEP, "") for w in (words or []) if isinstance(w, str)])
        out.append(rows)
    return out


def audio_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def aliyah_tokens(wlc: list, begin: str, end: str) -> list:
    """Ordered tokens for a verse range, one entry per timestamp.

    Every token is emitted (nothing is filtered) so the list stays index-aligned
    with the label file; unusable tokens are dropped later, after the zip.
    """
    bc, bv = (int(x) for x in begin.split(":"))
    ec, ev = (int(x) for x in end.split(":"))
    tokens = []
    for ch in range(bc, ec + 1):
        verses = wlc[ch - 1]
        lo = bv if ch == bc else 1
        hi = ev if ch == ec else len(verses)
        for v in range(lo, hi + 1):
            raw_words = verses[v - 1]
            verse_tokens = []
            for raw in raw_words:
                taamim = [TAAM_MAP[c] for c in unicodedata.normalize("NFC", raw) if c in TAAM_MAP]
                verse_tokens.append({
                    "raw": raw,
                    "norm": normalize_word(raw),
                    "taamim": taamim,
                    "taam": taamim[-1] if taamim else None,
                    "verse_ref": f"{ch}:{v}",
                })
            # Silluq is encoded with the meteg codepoint, so TAAM_MAP cannot see
            # it — the verse-final word carries it positionally.
            if verse_tokens and verse_tokens[-1]["taam"] is None:
                verse_tokens[-1]["taam"] = "Siluk"
                verse_tokens[-1]["taamim"] = verse_tokens[-1]["taamim"] + ["Siluk"]
            tokens.extend(verse_tokens)
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    ap.add_argument("--min-dur", type=float, default=0.1)
    ap.add_argument("--max-dur", type=float, default=3.0)
    args = ap.parse_args()

    if not ALIYAH_JSON.exists():
        sys.exit(f"missing {ALIYAH_JSON} — clone rneiss/PocketTorah into data/_sources/")

    with open(ALIYAH_JSON, encoding="utf-8-sig") as f:
        parshiot = json.load(f)["parshiot"]["parsha"]

    labels = index_dir(LABELS_DIR, ".txt")
    audios = index_dir(AUDIO_DIR, ".mp3")
    wlc_cache = {b: load_wlc(b) for b in BOOKS}

    records, stats = [], Counter()
    mismatches, missing = [], []
    per_parasha = {}

    print(f"{'parasha':<20}{'al':>3}{'range':>14}{'stamps':>8}{'words':>7}{'diff':>6}  status")
    print("-" * 74)

    for p in parshiot:
        pid, k = p["_id"], key(p["_id"])
        book = p["_verse"].split()[0]
        if book not in BOOKS:
            continue
        for a in p["fullkriyah"]["aliyah"]:
            try:
                num = int(a["_num"])
            except ValueError:
                continue                       # maftir entries are numbered 'M'
            lab, aud = labels.get((k, num)), audios.get((k, num))
            if not lab or not aud:
                missing.append((pid, num, "labels" if not lab else "audio"))
                stats["missing_file"] += 1
                continue

            ts = [float(x) for x in lab.read_text().strip().split(",")]
            toks = aliyah_tokens(wlc_cache[book], a["_begin"], a["_end"])
            diff = len(ts) - len(toks)
            rng = f"{a['_begin']}-{a['_end']}"

            if diff != 0:
                mismatches.append((pid, num, len(ts), len(toks), diff))
                stats["mismatched_aliyah"] += 1
                print(f"{pid:<20}{num:>3}{rng:>14}{len(ts):>8}{len(toks):>7}{diff:>+6}  SKIP")
                continue

            dur = audio_duration(aud)
            parasha_key = f"{k}_{num}"
            audio_file = str(aud.relative_to(PROJECT_DIR))
            kept = 0
            for i, w in enumerate(toks):
                if w["taam"] is None:
                    stats["dropped_no_taam"] += 1      # non-final maqaf-chain member
                    continue
                start = ts[i]
                end = ts[i + 1] if i + 1 < len(ts) else dur
                if not (args.min_dur <= end - start <= args.max_dur):
                    stats["dropped_bad_duration"] += 1
                    continue
                records.append({
                    "reader": READER,
                    "parasha": parasha_key,
                    "audio_file": audio_file,
                    "word_clean": w["norm"],
                    "word_raw": w["raw"],          # vocalized+cantillated, for syllable work
                    "taam": w["taam"],
                    "taamim_all": w["taamim"],
                    "verse_ref": f"{book} {w['verse_ref']}",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "probability": 1.0,            # human-verified, not ASR-estimated
                    "match": "gold",
                })
                kept += 1
                stats["verified"] += 1
            stats["aligned_aliyah"] += 1
            per_parasha[parasha_key] = {"stamps": len(ts), "words": len(toks), "kept": kept}
            print(f"{pid:<20}{num:>3}{rng:>14}{len(ts):>8}{len(toks):>7}{diff:>+6}  ok ({kept} kept)")

    print("-" * 74)
    n_al = stats["aligned_aliyah"] + stats["mismatched_aliyah"]
    print(f"aliyot aligned : {stats['aligned_aliyah']}/{n_al} "
          f"({100 * stats['aligned_aliyah'] / max(n_al, 1):.1f}% exact token match)")
    print(f"words verified : {stats['verified']:,}")
    print(f"dropped        : no ta'am {stats['dropped_no_taam']:,} · "
          f"bad duration {stats['dropped_bad_duration']:,}")
    if missing:
        print(f"missing files  : {len(missing)}  e.g. {missing[:5]}")
    if mismatches:
        print(f"token mismatch : {len(mismatches)}")
        for m in mismatches[:10]:
            print(f"    {m[0]} aliyah {m[1]}: {m[2]} stamps vs {m[3]} words ({m[4]:+d})")

    taam_counts = Counter(r["taam"] for r in records)
    print(f"\nta'amim ({len(taam_counts)} classes):")
    for t, c in taam_counts.most_common():
        print(f"    {t:<18}{c:>7,}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    out_dir = CLEANED_DIR / READER
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump({"reader": READER,
                   "source": "PocketTorah (github.com/rneiss/PocketTorah)",
                   "text": "Westminster Leningrad Codex (bundled with PocketTorah)",
                   "alignment": "human-verified word onsets, not ASR",
                   "stats": dict(stats), "per_parasha": per_parasha},
                  f, ensure_ascii=False, indent=1)
    print(f"\nwrote {len(records):,} records -> {OUT_JSON.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
