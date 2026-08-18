"""
Merge the per-source aligned datasets into one, and report the corpus design.
=============================================================================
  aligned_dataset_v2.json   R1..R5  Whisper timestamps, aligned against Sefaria
  aligned_dataset_pt.json   R6      PocketTorah gold onsets, WLC text

Readers whose audio is gone are dropped by default: a reader contributes a LORO
fold, and a fold built on one short recording reports noise as if it were a
generalization score. `--keep-all` overrides.

Usage:
  python Data_Prep/merge_datasets.py --report          # composition only
  python Data_Prep/merge_datasets.py                   # write merged dataset
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCES = ["aligned_dataset_v2.json", "aligned_dataset_pt.json"]
OUT = PROJECT_DIR / "aligned_dataset_merged.json"

# A reader needs enough coverage to stand as a held-out fold.
MIN_RECORDINGS = 5


def load_all(sources):
    records = []
    for name in sources:
        path = PROJECT_DIR / name
        if not path.exists():
            print(f"  missing, skipped: {name}")
            continue
        with open(path, encoding="utf-8") as f:
            rs = json.load(f)
        print(f"  {name}: {len(rs):,} records")
        records.extend(rs)
    return records


def audio_present(record) -> bool:
    return (PROJECT_DIR / record["audio_file"]).exists()


def report(records):
    by_reader = defaultdict(list)
    for r in records:
        by_reader[r["reader"]].append(r)

    print(f"\n{'reader':<8}{'words':>10}{'recordings':>12}{'audio on disk':>15}"
          f"{'alignment':>12}{'fold?':>8}")
    print("-" * 65)
    keep = []
    for reader in sorted(by_reader):
        rs = by_reader[reader]
        files = {r["audio_file"] for r in rs}
        present = {f for f in files if (PROJECT_DIR / f).exists()}
        kind = "gold" if rs[0].get("match") == "gold" else "whisper"
        usable = len(present) >= MIN_RECORDINGS
        keep.append((reader, usable))
        print(f"{reader:<8}{len(rs):>10,}{len(files):>12}{len(present):>10}/{len(files):<4}"
              f"{kind:>12}{'yes' if usable else 'NO':>8}")
    print("-" * 65)
    print(f"{'total':<8}{len(records):>10,}{len({r['audio_file'] for r in records}):>12}")
    return keep


def taam_table(records, readers):
    per = {r: Counter() for r in readers}
    for rec in records:
        if rec["reader"] in per:
            per[rec["reader"]][rec["taam"]] += 1
    all_taamim = Counter()
    for c in per.values():
        all_taamim.update(c)

    header = "ta'am"
    print(f"\n{header:<18}" + "".join(f"{r:>10}" for r in readers)
          + f"{'total':>10}{'min/max':>10}")
    print("-" * (18 + 10 * (len(readers) + 2)))
    for taam, total in all_taamim.most_common():
        counts = [per[r][taam] for r in readers]
        ratio = (min(counts) / max(counts)) if max(counts) else 0
        print(f"{taam:<18}" + "".join(f"{c:>10,}" for c in counts)
              + f"{total:>10,}{ratio:>10.2f}")
    print("-" * (18 + 10 * (len(readers) + 2)))
    print(f"{'classes':<18}{len(all_taamim):>10}")
    print("\nmin/max is the count ratio between the thinnest and richest reader for that\n"
          "ta'am — values far below 1 mean a held-out fold sees a class the training\n"
          "readers barely cover.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print composition, write nothing")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep readers with too few recordings to form a fold")
    ap.add_argument("--require-audio", action="store_true",
                    help="drop records whose audio file is not on disk")
    args = ap.parse_args()

    print("loading:")
    records = load_all(SOURCES)
    if not records:
        return

    keep = report(records)
    usable = {r for r, ok in keep if ok}
    dropped = {r for r, ok in keep if not ok}

    if not args.keep_all and dropped:
        before = len(records)
        records = [r for r in records if r["reader"] in usable]
        print(f"\ndropped readers with < {MIN_RECORDINGS} recordings on disk: "
              f"{sorted(dropped)} ({before - len(records):,} words)")

    if args.require_audio:
        before = len(records)
        records = [r for r in records if audio_present(r)]
        print(f"dropped records without audio: {before - len(records):,}")

    readers = sorted({r["reader"] for r in records})
    taam_table(records, readers)

    print(f"\nLORO folds available: {len(readers)} ({', '.join(readers)})")
    if len(readers) < 3:
        print("  WARNING: fewer than 3 readers — leave-one-reader-out is not a "
              "meaningful generalization test at this size.")

    if args.report:
        print("\n--report: nothing written")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\nwrote {len(records):,} records -> {OUT.name}")


if __name__ == "__main__":
    main()
