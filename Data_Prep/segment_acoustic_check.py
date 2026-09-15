"""
Automated per-segment sanity checks -- the closest thing to "listening"
that doesn't require a human ear.
=================================================================
Two checks, run over every one of the 114+ cut segments in
data/ben13_segments/manifest.json (every type, every taam):

1. EDGE TRUNCATION (pure signal processing, no ML): a clean cut tapers to
   near-silence at its boundary; a cut that sliced through live singing
   doesn't. Compare each edge's RMS energy against the segment's own peak.
   This is the direct automated version of "תבדוק שלא יהיו חתיכות" --
   cheap, deterministic, and does not depend on trusting a model's guess
   at what word is playing.

2. WHISPER COHERENCE (a real but limited proxy for "does this sound like
   speech"): run mlx-whisper on the ISOLATED segment (not the parent
   clip). An empty or near-empty result is a strong signal the clip is
   silence/noise/an artifact, not a spoken ta'am name. A non-empty result
   is NOT proof the word is correct -- Whisper doesn't know cantillation
   naming and mis-hears these compound names constantly (see
   whisper_crosscheck_ben13.py's findings) -- so this only catches the
   "nothing here" failure mode, not "wrong word" or "sounds bad."

Neither check is "hearing whether it sounds like Pashta." Nothing in this
toolchain can do that without a human ear -- that limitation is reported
honestly in results/v3_comparison/ben13_segment_review.md rather than
worked around with a check that overclaims what it verified.

Usage:
  python Data_Prep/segment_acoustic_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "data" / "ben13_segments" / "manifest.json"
OUT = PROJECT_DIR / "data" / "ben13_segments" / "acoustic_check.json"

EDGE_WINDOW = 0.03      # seconds examined at each boundary
TRUNC_RATIO = 0.35      # edge RMS above this fraction of peak RMS -> flagged


def edge_truncation(y: np.ndarray, sr: int):
    """(start_flag, end_flag, start_ratio, end_ratio)."""
    n = max(1, int(EDGE_WINDOW * sr))
    peak = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9
    # peak measured over the whole clip's RMS, not a single sample, so a
    # naturally loud recording doesn't bias the ratio
    peak = max(peak, float(np.max(np.abs(y))) * 0.3 + 1e-9)
    start_rms = float(np.sqrt(np.mean(y[:n].astype(np.float64) ** 2)))
    end_rms = float(np.sqrt(np.mean(y[-n:].astype(np.float64) ** 2)))
    sr_ratio, er_ratio = start_rms / peak, end_rms / peak
    return sr_ratio > TRUNC_RATIO, er_ratio > TRUNC_RATIO, round(sr_ratio, 3), round(er_ratio, 3)


def main():
    import mlx_whisper

    with open(MANIFEST, encoding="utf-8") as f:
        segments = json.load(f)

    results = {}
    for i, e in enumerate(segments, 1):
        path = PROJECT_DIR / e["local_path"]
        clip_id = Path(e["local_path"]).stem
        if not path.exists():
            continue
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)

        start_trunc, end_trunc, sr_ratio, er_ratio = edge_truncation(y, sr)

        out = mlx_whisper.transcribe(
            str(path), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language="he", word_timestamps=False)
        whisper_text = (out.get("text") or "").strip()

        flag = []
        if start_trunc:
            flag.append("start")
        if end_trunc:
            flag.append("end")
        if not whisper_text:
            flag.append("silent")

        results[clip_id] = {
            "taam": e["taam"], "source_type": e["source_type"],
            "source_order": e["source_order"],
            "start_edge_ratio": sr_ratio, "end_edge_ratio": er_ratio,
            "start_truncated": start_trunc, "end_truncated": end_trunc,
            "whisper_text": whisper_text, "whisper_empty": not whisper_text,
            "flags": flag,
        }
        if i % 20 == 0:
            print(f"  {i}/{len(segments)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    n_trunc = sum(1 for v in results.values() if v["start_truncated"] or v["end_truncated"])
    n_silent = sum(1 for v in results.values() if v["whisper_empty"])
    print(f"\n{len(results)} segments checked")
    print(f"  {n_trunc} flagged for edge truncation (start or end)")
    print(f"  {n_silent} flagged as silent/no speech detected")
    print(f"-> {OUT.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
