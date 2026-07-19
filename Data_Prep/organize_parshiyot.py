"""
Script to organize Torah reading audio files into R1 and R2 folders.
Checks for missing parshiyot and can download them from YouTube playlists.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

# Standard 54 parshiyot in order
PARSHIYOT = [
    ("01", "bereshit", "בראשית"),
    ("02", "noah", "נח"),
    ("03", "lech_lecha", "לך לך"),
    ("04", "vayera", "וירא"),
    ("05", "chayei_sarah", "חיי שרה"),
    ("06", "toledot", "תולדות"),
    ("07", "vayetze", "ויצא"),
    ("08", "vayishlach", "וישלח"),
    ("09", "vayeshev", "וישב"),
    ("10", "miketz", "מקץ"),
    ("11", "vayigash", "ויגש"),
    ("12", "vayechi", "ויחי"),
    ("13", "shemot", "שמות"),
    ("14", "vaera", "וארא"),
    ("15", "bo", "בא"),
    ("16", "beshalach", "בשלח"),
    ("17", "yitro", "יתרו"),
    ("18", "mishpatim", "משפטים"),
    ("19", "terumah", "תרומה"),
    ("20", "tetzaveh", "תצוה"),
    ("21", "ki_tisa", "כי תשא"),
    ("22", "vayakhel", "ויקהל"),
    ("23", "pekudei", "פקודי"),
    ("24", "vayikra", "ויקרא"),
    ("25", "tzav", "צו"),
    ("26", "shemini", "שמיני"),
    ("27", "tazria", "תזריע"),
    ("28", "metzora", "מצורע"),
    ("29", "acharei_mot", "אחרי מות"),
    ("30", "kedoshim", "קדושים"),
    ("31", "emor", "אמר"),
    ("32", "behar", "בהר"),
    ("33", "bechukotai", "בחוקותי"),
    ("34", "bamidbar", "במדבר"),
    ("35", "naso", "נסו"),
    ("36", "behaalotecha", "בהעלותך"),
    ("37", "shlach", "שלח"),
    ("38", "korach", "קורח"),
    ("39", "chukat", "חקת"),
    ("40", "balak", "בלק"),
    ("41", "pinchas", "פנחס"),
    ("42", "matot", "מטות"),
    ("43", "masei", "מסעי"),
    ("44", "devarim", "דברים"),
    ("45", "vaetchanan", "ואתחנן"),
    ("46", "eikev", "עקב"),
    ("47", "reeh", "ראה"),
    ("48", "shoftim", "שופטים"),
    ("49", "ki_teitzei", "כי תצא"),
    ("50", "ki_tavo", "כי תבוא"),
    ("51", "nitzavim", "נצבים"),
    ("52", "vayelech", "וילך"),
    ("53", "haazinu", "האזינו"),
    ("54", "vezot_haberacha", "וזאת הברכה"),
]

# Build lookup dicts
NUM_TO_NAME = {num: name for num, name, _ in PARSHIYOT}
NAME_TO_NUM = {name: num for num, name, _ in PARSHIYOT}
HEBREW_TO_NUM = {heb: num for num, _, heb in PARSHIYOT}


def get_existing_files(folder: Path):
    """Get .wav files in folder, return dict mapping parsha_num -> filename."""
    files = {}
    hebrew_files = []
    if not folder.exists():
        return files, hebrew_files
    for f in folder.iterdir():
        if f.suffix.lower() != ".wav":
            continue
        name = f.stem
        # Match numbered format: 01_bereshit
        m = re.match(r"(\d{2})_([a-z_]+)", name)
        if m:
            num, pname = m.groups()
            files[num] = str(f.name)
        else:
            hebrew_files.append(str(f.name))
    return files, hebrew_files


def analyze_reader(folder: Path, reader_name: str):
    print(f"\n{'='*60}")
    print(f"  {reader_name} - {folder}")
    print(f"{'='*60}")
    files, hebrew = get_existing_files(folder)
    print(f"  Numbered files: {len(files)}/54")
    if hebrew:
        print(f"  Hebrew-named files: {len(hebrew)}")
        for h in hebrew:
            print(f"    - {h}")

    missing = []
    for num, name, heb in PARSHIYOT:
        if num not in files:
            missing.append((num, name, heb))

    if missing:
        print(f"\n  Missing ({len(missing)}):")
        for num, name, heb in missing:
            print(f"    {num}_{name}  ({heb})")
    else:
        print("\n  Complete! No missing parshiyot.")

    return files, hebrew, missing


def rename_hebrew_files(folder: Path, hebrew_files: list, dry_run=True):
    """Try to rename Hebrew-named files to numbered format."""
    renamed = []
    for fname in hebrew_files:
        fpath = folder / fname
        # Try to identify parsha from Hebrew name in filename
        matched = False
        for num, ename, heb in PARSHIYOT:
            # Simple matching: check if Hebrew name or English name appears
            heb_no_spaces = heb.replace(" ", "")
            if heb in fname or heb_no_spaces in fname or ename in fname.lower():
                new_name = f"{num}_{ename}.wav"
                new_path = folder / new_name
                if new_path.exists():
                    print(f"  SKIP: {fname} -> {new_name} (target already exists)")
                else:
                    action = "WOULD RENAME" if dry_run else "RENAMED"
                    print(f"  {action}: {fname} -> {new_name}")
                    if not dry_run:
                        shutil.move(str(fpath), str(new_path))
                    renamed.append((fname, new_name))
                matched = True
                break
        if not matched:
            print(f"  COULD NOT IDENTIFY: {fname}")
    return renamed


def main():
    base = Path("g:/My Drive/02_Academia/כריית מידע/Deep Learning/Project")
    content_r1 = base / "content" / "R1"
    content_r2 = base / "content" / "R2"
    data_dvir = base / "data" / "dvir"
    data_r2 = base / "data" / "reader2"

    print("TORAH PARSHA ORGANIZER")
    print("=" * 60)

    # Analyze both readers
    r1_files, r1_hebrew, r1_missing = analyze_reader(content_r1, "Reader 1 (Dvir Porati)")
    r2_files, r2_hebrew, r2_missing = analyze_reader(content_r2, "Reader 2 (Study & Chumashim)")

    print(f"\n{'='*60}")
    print("  RENAMING HEBREW-NAMED FILES")
    print(f"{'='*60}")
    print("\n  R1 Hebrew files:")
    rename_hebrew_files(content_r1, r1_hebrew, dry_run=False)
    print("\n  R2 Hebrew files:")
    rename_hebrew_files(content_r2, r2_hebrew, dry_run=False)

    # Re-analyze after renaming
    r1_files, r1_hebrew, r1_missing = analyze_reader(content_r1, "Reader 1 (Dvir Porati)")
    r2_files, r2_hebrew, r2_missing = analyze_reader(content_r2, "Reader 2 (Study & Chumashim)")

    print(f"\n{'='*60}")
    print("  SUMMARY AFTER RENAMING")
    print(f"{'='*60}")
    print(f"  R1 missing: {len(r1_missing)} parshiyot")
    print(f"  R2 missing: {len(r2_missing)} parshiyot")

    # Cross-reference with data folders and copy
    print(f"\n{'='*60}")
    print("  COPYING FROM DATA FOLDERS")
    print(f"{'='*60}")
    for reader, missing, data_folder, content_folder in [
        ("R1", r1_missing, data_dvir, content_r1),
        ("R2", r2_missing, data_r2, content_r2),
    ]:
        if not missing:
            continue
        data_files, _ = get_existing_files(data_folder)
        copied = []
        for num, name, heb in missing:
            if num in data_files:
                src = data_folder / data_files[num]
                dst_name = f"{num}_{name}.wav"
                dst = content_folder / dst_name
                if not dst.exists():
                    print(f"  COPY {reader}: {src.name} -> {dst_name}")
                    shutil.copy2(str(src), str(dst))
                    copied.append((num, name, heb))
                else:
                    print(f"  SKIP {reader}: {dst_name} already exists")
        if copied:
            print(f"  {reader} - Copied {len(copied)} files")

    # Final analysis
    r1_files, r1_hebrew, r1_missing = analyze_reader(content_r1, "Reader 1 (Dvir Porati)")
    r2_files, r2_hebrew, r2_missing = analyze_reader(content_r2, "Reader 2 (Study & Chumashim)")

    print(f"\n{'='*60}")
    print("  FINAL STATUS")
    print(f"{'='*60}")
    print(f"  R1: {len(r1_files)}/54  |  Missing: {len(r1_missing)}")
    print(f"  R2: {len(r2_files)}/54  |  Missing: {len(r2_missing)}")

    all_missing = r1_missing + r2_missing
    if all_missing:
        print(f"\n  TOTAL MISSING TO DOWNLOAD: {len(all_missing)} unique parshiyot")
        print("  (Note: some may be missing from both readers)")
        for num, name, heb in all_missing:
            readers = []
            if any(n == num for n, _, _ in r1_missing):
                readers.append("R1")
            if any(n == num for n, _, _ in r2_missing):
                readers.append("R2")
            print(f"    {num}_{name} ({heb}) - needs in: {', '.join(readers)}")


if __name__ == "__main__":
    main()
