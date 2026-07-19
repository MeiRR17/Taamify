"""
Download missing parshiyot from YouTube playlists.
Uses yt-dlp to fetch playlist entries and download specific videos.
"""

import os
import re
import json
import subprocess
from pathlib import Path

# Standard 54 parshiyot
PARSHIYOT = [
    ("01", "bereshit", ["בראשית", "bereshit"]),
    ("02", "noah", ["נח", "noah", "noach"]),
    ("03", "lech_lecha", ["לך לך", "lech lecha"]),
    ("04", "vayera", ["וירא", "vayera"]),
    ("05", "chayei_sarah", ["חיי שרה", "chayei sarah"]),
    ("06", "toledot", ["תולדות", "toledot", "toldot"]),
    ("07", "vayetze", ["ויצא", "vayetze"]),
    ("08", "vayishlach", ["וישלח", "vayishlach"]),
    ("09", "vayeshev", ["וישב", "vayeshev"]),
    ("10", "miketz", ["מקץ", "miketz"]),
    ("11", "vayigash", ["ויגש", "vayigash"]),
    ("12", "vayechi", ["ויחי", "vayechi"]),
    ("13", "shemot", ["שמות", "shemot"]),
    ("14", "vaera", ["וארא", "vaera"]),
    ("15", "bo", ["בא", " bo ", "bo " ]),
    ("16", "beshalach", ["בשלח", "beshalach"]),
    ("17", "yitro", ["יתרו", "yitro"]),
    ("18", "mishpatim", ["משפטים", "mishpatim"]),
    ("19", "terumah", ["תרומה", "terumah"]),
    ("20", "tetzaveh", ["תצוה", "tetzaveh"]),
    ("21", "ki_tisa", ["כי תשא", "ki tisa"]),
    ("22", "vayakhel", ["ויקהל", "vayakhel"]),
    ("23", "pekudei", ["פקודי", "pekudei"]),
    ("24", "vayikra", ["ויקרא", "vayikra"]),
    ("25", "tzav", ["צו", " tzav ", "tzav "]),
    ("26", "shemini", ["שמיני", "shemini"]),
    ("27", "tazria", ["תזריע", "tazria"]),
    ("28", "metzora", ["מצורע", "metzora"]),
    ("29", "acharei_mot", ["אחרי מות", "acharei mot"]),
    ("30", "kedoshim", ["קדושים", "kedoshim"]),
    ("31", "emor", ["אמר", " emor ", "emor "]),
    ("32", "behar", ["בהר", "behar"]),
    ("33", "bechukotai", ["בחוקותי", "bechukotai"]),
    ("34", "bamidbar", ["במדבר", "bamidbar"]),
    ("35", "naso", ["נסו", "naso"]),
    ("36", "behaalotecha", ["בהעלותך", "behaalotecha"]),
    ("37", "shlach", ["שלח", "shlach"]),
    ("38", "korach", ["קורח", "korach"]),
    ("39", "chukat", ["חקת", "chukat"]),
    ("40", "balak", ["בלק", "balak"]),
    ("41", "pinchas", ["פנחס", "pinchas"]),
    ("42", "matot", ["מטות", "matot"]),
    ("43", "masei", ["מסעי", "masei"]),
    ("44", "devarim", ["דברים", "devarim"]),
    ("45", "vaetchanan", ["ואתחנן", "vaetchanan"]),
    ("46", "eikev", ["עקב", "eikev"]),
    ("47", "reeh", ["ראה", "reeh"]),
    ("48", "shoftim", ["שופטים", "shoftim"]),
    ("49", "ki_teitzei", ["כי תצא", "ki teitzei"]),
    ("50", "ki_tavo", ["כי תבוא", "ki tavo"]),
    ("51", "nitzavim", ["נצבים", "nitzavim"]),
    ("52", "vayelech", ["וילך", "vayelech"]),
    ("53", "haazinu", ["האזינו", "haazinu"]),
    ("54", "vezot_haberacha", ["וזאת הברכה", "vezot haberacha"]),
]

NAME_TO_NUM = {name: num for num, name, _ in PARSHIYOT}
NUM_TO_NAME = {num: name for num, name, _ in PARSHIYOT}


def run_ytdlp(args):
    """Run yt-dlp via python module."""
    cmd = ["python", "-m", "yt_dlp"] + args
    print(f"  Running: {' '.join(cmd[:10])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result


def get_playlist_items(playlist_url):
    """Get all video titles and IDs from a playlist."""
    print(f"\n  Fetching playlist: {playlist_url}")
    args = [
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-items", "1-100",
        playlist_url
    ]
    result = run_ytdlp(args)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return []
    try:
        data = json.loads(result.stdout)
        entries = data.get("entries", [])
        items = []
        for entry in entries:
            if entry:
                items.append({
                    "id": entry.get("id"),
                    "title": entry.get("title", ""),
                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}"
                })
        print(f"  Found {len(items)} videos")
        return items
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return []


def match_parsha(title, keywords):
    """Check if title matches parsha keywords."""
    title_lower = title.lower()
    for kw in keywords:
        if kw.lower() in title_lower:
            return True
    # Also check Hebrew
    for kw in keywords:
        if kw in title:
            return True
    return False


def download_audio(video_url, output_path):
    """Download audio from YouTube video as wav."""
    output_template = str(output_path.with_suffix(""))
    args = [
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", output_template + ".%(ext)s",
        "--no-playlist",
        video_url
    ]
    result = run_ytdlp(args)
    if result.returncode != 0:
        print(f"  Download error: {result.stderr[:500]}")
        return False
    # yt-dlp might produce .wav or .webm -> converted
    # Check if file exists
    for ext in [".wav", ".webm", ".m4a"]:
        f = output_path.with_suffix(ext)
        if f.exists():
            if ext != ".wav":
                # Convert to wav using ffmpeg if available, else rename
                wav_path = output_path.with_suffix(".wav")
                if not wav_path.exists():
                    conv = subprocess.run(
                        ["ffmpeg", "-i", str(f), "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(wav_path)],
                        capture_output=True
                    )
                    if conv.returncode == 0:
                        f.unlink()
                        print(f"  Converted to {wav_path.name}")
                    else:
                        print(f"  ffmpeg not available, keeping {f.name}")
            return True
    return False


def main():
    base = Path("g:/My Drive/02_Academia/כריית מידע/Deep Learning/Project")
    content_r1 = base / "content" / "R1"
    content_r2 = base / "content" / "R2"

    # Sources
    r1_playlist = "https://www.youtube.com/playlist?list=PLG4SNJhlUwMw-J1dWHaabX9J963BHP3tC"
    r2_playlists = [
        "https://www.youtube.com/playlist?list=PLIPrspE62cFLDZJLG7re1t3bMnTjfHRyU",  # בראשית
        "https://www.youtube.com/playlist?list=PLIPrspE62cFI7FC9jmPMucpOUYJqBsuNr",  # שמות
        "https://www.youtube.com/playlist?list=PLIPrspE62cFIs7iF_sF8IZfqGf2M1mY0x",  # ויקרא
        "https://www.youtube.com/playlist?list=PLIPrspE62cFJiJIM30ZhDORbr9KLvKIrE",  # במדבר
        "https://www.youtube.com/playlist?list=PLIPrspE62cFKbEFx4-BD6t-KvE8J4ns90",  # דברים
    ]

    # Define what's missing
    r1_missing_nums = ["01", "05", "11", "28", "33", "41", "43", "54"]
    r2_missing_nums = ["28", "41"]

    print("=" * 70)
    print("  DOWNLOAD MISSING PARSHIYOT")
    print("=" * 70)

    # --- R1 Downloads ---
    print("\n" + "=" * 70)
    print("  READER 1 (Dvir Porati)")
    print("  Playlist:", r1_playlist)
    print("=" * 70)

    r1_items = get_playlist_items(r1_playlist)
    if r1_items:
        # Map missing parshiyot to videos
        for num, name, keywords in PARSHIYOT:
            if num not in r1_missing_nums:
                continue
            print(f"\n  Looking for {num}_{name}...")
            matched = None
            for item in r1_items:
                if match_parsha(item["title"], keywords):
                    matched = item
                    break
            if matched:
                print(f"    MATCH: {matched['title']}")
                output_path = content_r1 / f"{num}_{name}.wav"
                if output_path.exists():
                    print(f"    Already exists, skipping")
                else:
                    print(f"    Downloading to {output_path.name}...")
                    success = download_audio(matched["url"], output_path)
                    if success:
                        print(f"    SUCCESS!")
                    else:
                        print(f"    FAILED")
            else:
                print(f"    NO MATCH found for {name}")
                print(f"    Available titles sample:")
                for item in r1_items[:5]:
                    print(f"      - {item['title']}")

    # --- R2 Downloads ---
    print("\n" + "=" * 70)
    print("  READER 2 (Study & Chumashim)")
    print("=" * 70)

    # Fetch all R2 playlists
    all_r2_items = []
    for pl in r2_playlists:
        items = get_playlist_items(pl)
        all_r2_items.extend(items)

    if all_r2_items:
        for num, name, keywords in PARSHIYOT:
            if num not in r2_missing_nums:
                continue
            print(f"\n  Looking for {num}_{name}...")
            matched = None
            for item in all_r2_items:
                if match_parsha(item["title"], keywords):
                    matched = item
                    break
            if matched:
                print(f"    MATCH: {matched['title']}")
                output_path = content_r2 / f"{num}_{name}.wav"
                if output_path.exists():
                    print(f"    Already exists, skipping")
                else:
                    print(f"    Downloading to {output_path.name}...")
                    success = download_audio(matched["url"], output_path)
                    if success:
                        print(f"    SUCCESS!")
                    else:
                        print(f"    FAILED")
            else:
                print(f"    NO MATCH found for {name}")
                print(f"    Available titles sample:")
                for item in all_r2_items[:5]:
                    print(f"      - {item['title']}")

    print("\n" + "=" * 70)
    print("  DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
