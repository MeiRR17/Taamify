"""
Word-level Whisper transcription on Apple Silicon (mlx-whisper).
================================================================
Re-transcribes recordings whose metadata lacks word timestamps
(R4/R5 bereshit files were transcribed without word_timestamps=True).

Output JSON matches the faster-whisper format the rest of the pipeline expects:
  {"text": ..., "segments": [{"start","end","text","words":[{"word","start","end","probability"}]}]}

Usage:
  python Data_Prep/transcribe_mac.py data/R4/wav/01_bereshit.wav metadata/R4/r4_01_bereshit.json
  python Data_Prep/transcribe_mac.py --all-missing        # scan metadata/ for files with 0 words
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL = "mlx-community/whisper-large-v3-mlx"


def transcribe(audio_path: Path, output_path: Path):
    import mlx_whisper

    print(f"Opening {audio_path.name} (downloads from Drive if not local — can take minutes)...",
          flush=True)
    t0 = time.time()
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=MODEL,
        language="he",
        word_timestamps=True,
        verbose=False,
    )
    elapsed = time.time() - t0

    segments = []
    for seg in result.get("segments", []):
        words = [
            {
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": float(w.get("probability", 0.0)),
            }
            for w in seg.get("words", [])
        ]
        segments.append({
            "id": seg.get("id"),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", ""),
            "words": words,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"text": result.get("text", ""), "segments": segments,
                   "language": result.get("language", "he")}, f, ensure_ascii=False, indent=1)

    n_words = sum(len(s["words"]) for s in segments)
    print(f"  Done in {elapsed/60:.1f} min: {len(segments)} segments, {n_words} words -> {output_path}",
          flush=True)
    if n_words == 0:
        print("  WARNING: no word timestamps produced!")
        return False
    return True


def find_missing():
    """Metadata JSONs that have segments but zero word-level timestamps."""
    missing = []
    for meta in sorted((PROJECT_DIR / "metadata").glob("R*/*.json")):
        try:
            with open(meta, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        n_words = sum(len(s.get("words", [])) for s in d.get("segments", []))
        if n_words == 0:
            reader = meta.parent.name
            stem = meta.stem
            for prefix in ("r1_", "r2_", "r3_", "r4_", "r5_"):
                stem = stem.replace(prefix, "", 1) if stem.startswith(prefix) else stem
            wav = PROJECT_DIR / "data" / reader / "wav" / f"{stem}.wav"
            missing.append((wav, meta))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="input wav")
    ap.add_argument("output", nargs="?", help="output metadata json")
    ap.add_argument("--all-missing", action="store_true",
                    help="re-transcribe every metadata file that lacks word timestamps")
    args = ap.parse_args()

    if args.all_missing:
        missing = find_missing()
        if not missing:
            print("No metadata files with missing word timestamps.")
            return
        print(f"{len(missing)} recordings need re-transcription:")
        for wav, meta in missing:
            print(f"  {wav} -> {meta}")
        ok = True
        for i, (wav, meta) in enumerate(missing, 1):
            if not wav.exists():
                print(f"  SKIP: {wav} not found", flush=True)
                ok = False
                continue
            print(f"\n[{i}/{len(missing)}]", flush=True)
            try:
                ok = transcribe(wav, meta) and ok
            except Exception as e:
                # transient Drive read / ffmpeg failure — keep going, a rerun
                # of --all-missing picks this file up again
                print(f"  ERROR on {wav.name}: {e} — continuing", flush=True)
                ok = False
        sys.exit(0 if ok else 1)

    if not (args.audio and args.output):
        ap.error("provide AUDIO and OUTPUT, or --all-missing")
    transcribe(Path(args.audio), Path(args.output))


if __name__ == "__main__":
    main()
