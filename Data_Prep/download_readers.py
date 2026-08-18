"""
Re-download the YouTube reader corpora (R1, R2) into data/<reader>/wav/.
=======================================================================
Replaces download_missing.py, which hardcoded a Windows Google-Drive path and
only chased the handful of parshiyot missing at the time.

Titles are matched to parasha keys on Hebrew name with word boundaries — the
old substring test misfires badly on the short names (בא, צו, אמר are
substrings of half the corpus).

Output: data/<reader>/wav/<NN>_<name>.wav — mono 44.1 kHz, the layout
clean_align_v2.py and extract_features_v2.py expect.

Usage:
  python Data_Prep/download_readers.py --dry-run        # show the title->parasha map
  python Data_Prep/download_readers.py                  # download everything missing
  python Data_Prep/download_readers.py --readers R1
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# (key, hebrew names). Hebrew spelling varies across channels, so several
# variants per parasha; matching is whole-word, not substring.
PARSHIYOT = [
    ("01_bereshit", ["בראשית"]),
    ("02_noah", ["נח"]),
    ("03_lech_lecha", ["לך לך", "לך-לך"]),
    ("04_vayera", ["וירא"]),
    ("05_chayei_sarah", ["חיי שרה", "חיי-שרה"]),
    ("06_toledot", ["תולדות", "תלדות"]),
    ("07_vayetze", ["ויצא"]),
    ("08_vayishlach", ["וישלח"]),
    ("09_vayeshev", ["וישב"]),
    ("10_miketz", ["מקץ"]),
    ("11_vayigash", ["ויגש"]),
    ("12_vayechi", ["ויחי"]),
    ("13_shemot", ["שמות"]),
    ("14_vaera", ["וארא"]),
    ("15_bo", ["בא"]),
    ("16_beshalach", ["בשלח"]),
    ("17_yitro", ["יתרו"]),
    ("18_mishpatim", ["משפטים"]),
    ("19_terumah", ["תרומה"]),
    ("20_tetzaveh", ["תצוה", "תצווה"]),
    ("21_ki_tisa", ["כי תשא", "כי-תשא"]),
    ("22_vayakhel", ["ויקהל"]),
    ("23_pekudei", ["פקודי"]),
    ("24_vayikra", ["ויקרא"]),
    ("25_tzav", ["צו"]),
    ("26_shemini", ["שמיני"]),
    ("27_tazria", ["תזריע"]),
    ("28_metzora", ["מצורע"]),
    ("29_acharei_mot", ["אחרי מות", "אחרי-מות"]),
    ("30_kedoshim", ["קדושים", "קדשים"]),
    ("31_emor", ["אמר", "אמור"]),
    ("32_behar", ["בהר"]),
    ("33_bechukotai", ["בחוקותי", "בחקתי", "בחוקתי"]),
    ("34_bamidbar", ["במדבר"]),
    ("35_naso", ["נשא", "נסו"]),
    ("36_behaalotecha", ["בהעלותך", "בהעלתך"]),
    ("37_shlach", ["שלח"]),
    ("38_korach", ["קורח", "קרח"]),
    ("39_chukat", ["חוקת", "חקת"]),
    ("40_balak", ["בלק"]),
    ("41_pinchas", ["פנחס", "פינחס"]),
    ("42_matot", ["מטות", "מטת"]),
    ("43_masei", ["מסעי"]),
    ("44_devarim", ["דברים"]),
    ("45_vaetchanan", ["ואתחנן"]),
    ("46_eikev", ["עקב", "עיקב"]),
    ("47_reeh", ["ראה"]),
    ("48_shoftim", ["שופטים", "שפטים"]),
    ("49_ki_teitzei", ["כי תצא", "כי-תצא"]),
    ("50_ki_tavo", ["כי תבוא", "כי-תבוא"]),
    ("51_nitzavim", ["נצבים", "ניצבים"]),
    ("52_vayelech", ["וילך"]),
    ("53_haazinu", ["האזינו"]),
    ("54_vezot_haberacha", ["וזאת הברכה", "וזאת-הברכה"]),
]

SOURCES = {
    # Dvir Porati — full Torah, Sephardic Jerusalem nusach
    "R1": ["https://www.youtube.com/playlist?list=PLG4SNJhlUwMw-J1dWHaabX9J963BHP3tC"],
    # Sephardic Jerusalem nusach, one playlist per chumash
    "R2": [
        "https://www.youtube.com/playlist?list=PLIPrspE62cFLDZJLG7re1t3bMnTjfHRyU",  # בראשית
        "https://www.youtube.com/playlist?list=PLIPrspE62cFI7FC9jmPMucpOUYJqBsuNr",  # שמות
        "https://www.youtube.com/playlist?list=PLIPrspE62cFIs7iF_sF8IZfqGf2M1mY0x",  # ויקרא
        "https://www.youtube.com/playlist?list=PLIPrspE62cFJiJIM30ZhDORbr9KLvKIrE",  # במדבר
        "https://www.youtube.com/playlist?list=PLIPrspE62cFKbEFx4-BD6t-KvE8J4ns90",  # דברים
    ],
}

# Some uploads are titled in English only, so the Hebrew pass finds nothing.
LATIN_ALIASES = {
    "35_naso": ["naso", "nasso"],
    "43_masei": ["masei", "mas'ei"],
    "37_shlach": ["shlach", "sh'lach", "shelach"],
    "36_behaalotecha": ["behaalotcha", "beha'alotcha"],
    "45_vaetchanan": ["vaetchanan", "va'ethanan", "vaethanan"],
    "47_reeh": ["re'eh", "reeh"],
}

HEB_BOUNDARY = r"(?:^|[^א-ת])"


def match_parasha(title: str):
    """Longest whole-word Hebrew match, so 'לך לך' wins over 'לך' and בא never
    matches inside בשלח. Falls back to the Latin title for English-only uploads."""
    best = None
    for key, names in PARSHIYOT:
        for name in names:
            pat = HEB_BOUNDARY + re.escape(name) + r"(?![א-ת])"
            if re.search(pat, title):
                if best is None or len(name) > best[1]:
                    best = (key, len(name))
    if best:
        return best[0]

    low = title.lower()
    for key, aliases in LATIN_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", low):
                return key
    return None


# YouTube answers anonymous requests with "Sign in to confirm you're not a bot",
# so every call carries browser cookies.
COOKIES = []


def playlist_items(url: str) -> list:
    r = subprocess.run(
        ["yt-dlp", *COOKIES, "--flat-playlist", "--dump-single-json",
         "--socket-timeout", "30", url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  ERROR listing {url}: {r.stderr.strip()[:300]}")
        return []
    entries = json.loads(r.stdout).get("entries", []) or []
    return [{"id": e.get("id"), "title": e.get("title", "")} for e in entries if e]


def download(video_id: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["yt-dlp", *COOKIES, "-f", "bestaudio/best",
         "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
         "--postprocessor-args", "ffmpeg:-ac 1 -ar 44100",
         "--no-playlist", "--retries", "10", "--fragment-retries", "20",
         # YouTube throttles bursts with intermittent 403s on the media URL;
         # pacing the requests converts most of those into successes.
         "--sleep-requests", "2", "--sleep-interval", "5", "--max-sleep-interval", "20",
         "-o", str(out_path.with_suffix("")) + ".%(ext)s",
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"      FAILED: {r.stderr.strip()[-300:]}")
        return False
    return out_path.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=list(SOURCES))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the title->parasha map, download nothing")
    ap.add_argument("--cookies-from-browser", default="chrome",
                    help="browser to pull YouTube cookies from ('none' to disable)")
    args = ap.parse_args()

    global COOKIES
    if args.cookies_from_browser and args.cookies_from_browser != "none":
        COOKIES = ["--cookies-from-browser", args.cookies_from_browser]

    grand = {}
    for reader in args.readers:
        if reader not in SOURCES:
            print(f"{reader}: no source URLs on record — skipping")
            continue
        print(f"\n{'=' * 70}\n  {reader}\n{'=' * 70}")
        items = []
        for url in SOURCES[reader]:
            items.extend(playlist_items(url))
        print(f"  {len(items)} videos listed")

        mapped, unmatched, dupes = {}, [], []
        for it in items:
            key = match_parasha(it["title"])
            if key is None:
                unmatched.append(it["title"])
            elif key in mapped:
                dupes.append((key, it["title"]))
            else:
                mapped[key] = it

        print(f"  matched {len(mapped)}/54 parshiyot"
              f"{f' · {len(unmatched)} unmatched' if unmatched else ''}"
              f"{f' · {len(dupes)} duplicate' if dupes else ''}")
        for t in unmatched:
            print(f"    UNMATCHED: {t[:90]}")
        for k, t in dupes:
            print(f"    DUPLICATE {k}: {t[:80]}")

        out_dir = DATA_DIR / reader / "wav"
        todo = [(k, v) for k, v in sorted(mapped.items())
                if not (out_dir / f"{k}.wav").exists()]
        have = len(mapped) - len(todo)
        print(f"  already on disk: {have} · to download: {len(todo)}")
        grand[reader] = {"matched": len(mapped), "todo": len(todo)}

        if args.dry_run:
            for k, v in sorted(mapped.items())[:60]:
                mark = "·" if (out_dir / f"{k}.wav").exists() else " "
                print(f"    {mark} {k:<22} {v['title'][:70]}")
            continue

        for i, (key, item) in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {key} <- {item['title'][:60]}")
            if not download(item["id"], out_dir / f"{key}.wav"):
                print(f"      could not fetch {key}")

    print(f"\n{'=' * 70}")
    for r, s in grand.items():
        print(f"  {r}: matched {s['matched']}/54, {s['todo']} pending")
    if args.dry_run:
        print("  --dry-run: nothing downloaded")


if __name__ == "__main__":
    main()
