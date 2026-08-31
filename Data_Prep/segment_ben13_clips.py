"""
Segment Ben13's combination clips into per-ta'am audio pieces.
================================================================
Ben13's combination clips (e.g. "Mercha Tipcha Munach Etnachta" as one
~5-second clip) are the teacher chanting each ta'am's own name in sequence,
sung with that ta'am's melody — confirmed by transcribing sample clips with
mlx-whisper and matching the output phonetically against the manifest's own
`text`/`textSfaradi` fields (see Milestone 1C in the plan). That means the
exact expected text for forced alignment is already in the manifest — no
Sefaria lookup needed, unlike every other alignment in this project.

Naive cuts don't work: most of a clip's duration is melismatic tail that
Whisper's own word timestamps truncate and that carries no silence to cut
on (confirmed with ffmpeg silencedetect/volumedetect — the "quiet" tail is
still full-volume singing). `verse_align.align_verse()` is the right tool —
it's a plain (waveform, word-list) -> (spans, score) forced alignment with
no Sefaria dependency, already proven on real chanted Hebrew.

Its MIN_SCORE/MAX_EDGE_GAP thresholds are calibrated for long multi-word
Torah verses, not 1-5 word phrases, so this script records every score as a
quality weight rather than hard-gating on it — consistent with how
verse_align.py itself treats scores. Flag (don't discard) low-confidence
cuts for manual review, especially rare-vocabulary clips (Yerach Ben Yomo /
Qarney Para) where even Whisper's ASR struggled.

Output:
  data/ben13_segments/<type>/<Taam>_<type>-<order>.wav
  data/ben13_segments/<type>/manifest.json
    [{taam, source_type, source_order, source_url, ctc_score, duration,
      local_path}]

Usage:
  python Data_Prep/segment_ben13_clips.py             # all types
  python Data_Prep/segment_ben13_clips.py --types teamimtora
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import TAAM_MAP, normalize_word  # noqa: E402
from ctc_align import SR, load_recording  # noqa: E402
from verse_align import align_verse  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEN13_DIR = PROJECT_DIR / "data" / "ben13"
OUT_DIR = PROJECT_DIR / "data" / "ben13_segments"

TYPES = ["teamimtora", "teamim-meg", "teamim-haf"]

# verse_align.MIN_SCORE (-3.2) was calibrated on long verses; these are
# 1-5 word phrases, a different regime. Flag well below that as suspect
# rather than silently trusting it, without hard-discarding anything.
FLAG_SCORE = -3.5


def token_taams(token: str) -> list:
    """Ta'amim decoded from one token's own diacritics, in order."""
    return [TAAM_MAP[ch] for ch in token if ch in TAAM_MAP]


def words_from_entry(entry: dict):
    """(words, taam_per_word) for forced alignment, or (None, None) if neither
    text field's tokenization matches taam_sequence 1:1 (some tokens carry
    two ta'amim in one word — e.g. "two Pashtas" as a single compound word —
    or a field is missing/incomplete; these need manual review, not a guess).

    A token with zero ta'amim (e.g. "סוף-פסוק", the end-of-verse marker some
    clips end on) is kept as a real word for alignment — the audio still
    speaks it — but tagged with taam=None so no segment is emitted for it,
    which also stops it from silently swallowing the true final ta'am's span
    (align_verse extends each word to the onset of the next).
    """
    seq = entry["taam_sequence"]
    for field in ("textSfaradi", "text"):
        raw = entry.get(field) or ""
        tokens = raw.split()
        if not tokens:
            continue
        per_token = [token_taams(t) for t in tokens]
        if any(len(g) > 1 for g in per_token):
            continue  # a word carrying >1 ta'am can't be split by word alignment
        flat = [g[0] for g in per_token if g]
        if flat != seq:
            continue
        words = [{"raw": t, "norm": normalize_word(t)} for t in tokens]
        taams = [g[0] if g else None for g in per_token]
        return words, taams
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=TYPES, choices=TYPES)
    args = ap.parse_args()

    manifest_out = []
    for type_name in args.types:
        manifest_path = BEN13_DIR / type_name / "manifest.json"
        if not manifest_path.exists():
            print(f"  {type_name}: no manifest — run download_ben13.py first")
            continue
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
        combos = [e for e in entries if len(e["taam_sequence"]) > 1]
        print(f"\n{'=' * 70}\n  {type_name}: {len(combos)} combination clips\n{'=' * 70}")

        out_dir = OUT_DIR / type_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for e in sorted(combos, key=lambda x: x["order"]):
            seq = e["taam_sequence"]
            words, taams = words_from_entry(e)
            if words is None:
                print(f"    SKIP order {e['order']}: no clean 1:1 word/ta'am "
                      f"tokenization in text or textSfaradi — needs manual "
                      f"review ('{e['text']}')")
                continue

            audio_path = PROJECT_DIR / e["local_path"]
            if not audio_path.exists():
                print(f"    SKIP order {e['order']}: {audio_path.name} not on "
                      f"disk (download_ben13.py reported this one 404'd)")
                continue
            y = load_recording(audio_path)
            spans, score, _ = align_verse(y, words)
            if spans is None:
                print(f"    SKIP order {e['order']}: forced alignment failed "
                      f"('{e['text']}')")
                continue

            flag = " <- LOW SCORE, review by hand" if score < FLAG_SCORE else ""
            print(f"    order {e['order']:>3} score {score:>6.2f}{flag}  "
                  f"{' -> '.join(seq)}")

            for i, cls in enumerate(taams):
                if cls is None:
                    continue
                start, end = spans[i]
                s_idx, e_idx = int(start * SR), int(end * SR)
                clip = y[s_idx:e_idx]
                if len(clip) < SR * 0.05:
                    continue
                # position i is part of the filename: a class can repeat
                # within one combo clip (e.g. "Mercha -> Tipecha -> Mercha"),
                # and without it the second occurrence silently overwrote the
                # first on disk -- confirmed after the fact via a 114-vs-102
                # unique-filename count mismatch against the manifest.
                local_name = f"{cls}_{type_name}-{e['order']}_{i}.wav"
                local_path = out_dir / local_name
                sf.write(local_path, clip, SR)
                manifest_out.append({
                    "taam": cls,
                    "source_type": type_name,
                    "source_order": e["order"],
                    "source_url": e["resolved_url"],
                    "ctc_score": round(score, 3),
                    "duration": round(len(clip) / SR, 3),
                    "local_path": str(local_path.relative_to(PROJECT_DIR)),
                    "low_confidence": score < FLAG_SCORE,
                })

    out_manifest = OUT_DIR / "manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, ensure_ascii=False, indent=1)

    print(f"\n{'=' * 70}")
    by_class = {}
    for item in manifest_out:
        by_class.setdefault(item["taam"], []).append(item)
    for cls, items in sorted(by_class.items()):
        n_low = sum(1 for i in items if i["low_confidence"])
        print(f"  {cls:<16} {len(items)} segment(s)"
              f"{f' ({n_low} low-confidence)' if n_low else ''}")
    print(f"\nwrote {len(manifest_out)} segments -> {out_manifest.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
