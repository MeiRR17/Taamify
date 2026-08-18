"""
Gate 2 check: is a new reader's alignment as trustworthy as the existing ones?
===============================================================================
Yemenite Hebrew is phonologically distinct from the modern Israeli Hebrew that
both Whisper and imvladikon/wav2vec2-xls-r-300m-hebrew were trained on -
different realisations of gimel, dalet, tav, and distinct vowel qualities. If
either model aligns this material worse, a low cross-tradition transfer score
would be an alignment artefact wearing the costume of a finding, not the
finding itself.

This compares the new reader's alignment statistics against R1/R2's on the same
measures already used to validate them: clean_align_v2's word survival rate,
and verse_align's rejection rates and mean forced-alignment score.

Usage:
  python Data_Prep/validate_new_reader.py --reader R7
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_DIR / "metadata_cleaned_v2"


def clean_align_stats(reader: str):
    total_whisper = total_verified = total_sefaria = 0
    n_files = 0
    for p in sorted((CLEANED_DIR / reader).glob("*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        s = d.get("stats", {})
        total_whisper += s.get("whisper_words", 0)
        total_verified += s.get("verified", 0)
        total_sefaria += s.get("sefaria_words", 0)
        n_files += 1
    return {"files": n_files, "whisper_words": total_whisper,
            "verified": total_verified, "sefaria_words": total_sefaria,
            "keep_pct": 100 * total_verified / max(total_whisper, 1),
            "coverage_pct": 100 * total_verified / max(total_sefaria, 1)}


def verse_align_stats(reader: str):
    parts_dir = Path.home() / ".cache" / "taamify" / "verse_parts"
    stats = {}
    scores = []
    for p in parts_dir.glob(f"data__{reader}__wav__*.json"):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.get("stats", {}).items():
            stats[k] = stats.get(k, 0) + v
        for w in d.get("windows", {}).values():
            if "score" in w:
                scores.append(w["score"])
    return stats, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--baseline", nargs="+", default=["R1", "R2"])
    args = ap.parse_args()

    print(f"{'reader':<8}{'files':>7}{'whisper words':>15}{'verified':>10}"
          f"{'keep %':>9}{'sefaria cov %':>15}")
    print("-" * 64)
    rows = {}
    for r in [args.reader] + args.baseline:
        s = clean_align_stats(r)
        rows[r] = s
        print(f"{r:<8}{s['files']:>7}{s['whisper_words']:>15,}{s['verified']:>10,}"
              f"{s['keep_pct']:>8.1f}%{s['coverage_pct']:>14.1f}%")
    print("-" * 64)

    new_keep = rows[args.reader]["keep_pct"]
    base_keep = sum(rows[b]["keep_pct"] for b in args.baseline) / len(args.baseline)
    print(f"\n{args.reader} keeps {new_keep:.1f}% of Whisper words vs "
          f"{base_keep:.1f}% average for {args.baseline}.")
    if new_keep < base_keep - 10:
        print("WARNING: markedly lower survival — the alignment may be struggling\n"
              "with this reader's phonology. Do not trust transfer numbers on it\n"
              "without investigating further (listen to ~30 windows by hand).")
    else:
        print("Comparable to the baseline readers — alignment looks trustworthy.")

    print(f"\n── verse-level CTC (if run) ──")
    for r in [args.reader] + args.baseline:
        stats, scores = verse_align_stats(r)
        if not stats:
            print(f"{r}: not yet run through verse_align.py")
            continue
        total = sum(stats.values())
        aligned = stats.get("aligned", 0)
        mean_score = sum(scores) / len(scores) if scores else float("nan")
        print(f"{r}: {aligned:,}/{total:,} aligned ({100*aligned/max(total,1):.1f}%), "
              f"mean score {mean_score:.2f}, rejections {dict(stats)}")


if __name__ == "__main__":
    main()
