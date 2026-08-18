"""
Fold proportional_verse_locate.py's output into the standard corpus format.
=============================================================================
proportional_verse_locate produces, per word, both the whole-word span
(word_start/word_end) and — where the CTC path let it compute one — the
stressed-syllable span, in one pass. This writes both into the places the rest
of the pipeline already knows how to read:

  aligned_dataset_<reader>.json   the record itself: reader, parasha, taam,
                                   verse_ref, start/end = the whole word.
                                   match="proportional_ctc" records provenance,
                                   same as every other source in this corpus.

  verse_aligned_windows.json      merged in under the same join key
                                   extract_features_v3 already uses
                                   (f"{reader}|{parasha}|{start}"), so both the
                                   word window and the stressed window paths
                                   pick up these acoustic boundaries instead of
                                   falling back to unset ("stored") ones.

Usage:
  python Data_Prep/ingest_proportional.py --reader R7
"""

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VERSE_WINDOWS = PROJECT_DIR / "verse_aligned_windows.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    args = ap.parse_args()

    src = PROJECT_DIR / f"proportional_windows_{args.reader}.json"
    with open(src, encoding="utf-8") as f:
        proportional = json.load(f)
    print(f"loaded {len(proportional):,} proportionally-aligned words")

    records = []
    verse_entries = {}
    for key, w in proportional.items():
        reader, parasha = key.split("|")[0], key.split("|")[1]
        audio_file = f"data/{reader}/wav/{parasha}.wav"
        records.append({
            "reader": reader, "parasha": parasha, "audio_file": audio_file,
            "word_clean": w["word_clean"], "word_raw": w["word_raw"],
            "taam": w["taam"], "taamim_all": w["taamim_all"],
            "verse_ref": w["verse_ref"],
            "start": w["word_start"], "end": w["word_end"],
            "probability": 1.0, "match": "proportional_ctc",
        })
        join_key = f"{reader}|{parasha}|{w['word_start']}"
        entry = {"word_start": w["word_start"], "word_end": w["word_end"],
                 "method": "proportional_ctc"}
        if "start" in w:
            entry["start"], entry["end"] = w["start"], w["end"]
        verse_entries[join_key] = entry

    out = PROJECT_DIR / f"aligned_dataset_{args.reader.lower()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"wrote {len(records):,} records -> {out.name}")

    existing = {}
    if VERSE_WINDOWS.exists():
        with open(VERSE_WINDOWS, encoding="utf-8") as f:
            existing = json.load(f)
    before = len(existing)
    existing.update(verse_entries)
    with open(VERSE_WINDOWS, "w", encoding="utf-8") as f:
        json.dump(existing, f)
    print(f"verse_aligned_windows.json: {before:,} -> {len(existing):,} "
          f"(+{len(existing) - before:,})")

    print(f"\nAdd '{out.name}' to SOURCES in Data_Prep/merge_datasets.py, then:\n"
          f"  python Data_Prep/merge_datasets.py --require-audio")


if __name__ == "__main__":
    main()
