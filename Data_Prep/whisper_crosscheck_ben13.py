"""
Independent cross-check: does Whisper hear the same word COUNT Ben13's
own text claims?
=================================================================
The manual review found Ben13's declared text sometimes lists words that
are never sung (see results/v3_comparison/ben13_segment_review.md). CTC
forced alignment can't catch that on its own -- handed too many words, it
compresses silently instead of failing.

This is not a replacement for the human reviewer (cut.html) -- Whisper
doesn't know Sephardic ta'am names, so it won't produce a usable
transcript on its own. What it IS good for: transcribing each clip with
NO constraint from Ben13's text at all, so its word count is an
independent second opinion. Declared count vs. Whisper's count agreeing
is weak evidence the text is trustworthy; disagreeing is a strong signal
to check that clip in cut.html first, catching cases a duration-ratio
heuristic alone would miss (right word count, wrong middle word).

Usage:
  python Data_Prep/whisper_crosscheck_ben13.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEN13_DIR = PROJECT_DIR / "data" / "ben13"
OUT = PROJECT_DIR / "data" / "ben13_segments" / "whisper_crosscheck.json"
MODEL = "mlx-community/whisper-large-v3-mlx"


def main():
    import mlx_whisper

    results = {}
    for type_name in ["teamimtora", "teamim-meg", "teamim-haf"]:
        mpath = BEN13_DIR / type_name / "manifest.json"
        if not mpath.exists():
            continue
        with open(mpath, encoding="utf-8") as f:
            entries = json.load(f)
        combos = [e for e in entries if len(e["taam_sequence"]) > 1]
        print(f"\n{type_name}: {len(combos)} combo clips")
        for e in sorted(combos, key=lambda x: x["order"]):
            audio_path = PROJECT_DIR / e["local_path"]
            if not audio_path.exists():
                continue
            out = mlx_whisper.transcribe(
                str(audio_path), path_or_hf_repo=MODEL, language="he",
                word_timestamps=True)
            words = [w["word"].strip() for seg in out["segments"]
                     for w in seg.get("words", []) if w["word"].strip()]
            declared_n = len(e["taam_sequence"])
            whisper_n = len(words)
            mismatch = whisper_n != declared_n
            flag = " <- COUNT MISMATCH" if mismatch else ""
            print(f"  order {e['order']:>3}  declared={declared_n}  "
                  f"whisper={whisper_n}{flag}   whisper text: {' '.join(words)}")
            results[f"{type_name}-{e['order']}"] = {
                "declared_sequence": e["taam_sequence"],
                "declared_n": declared_n,
                "whisper_words": words,
                "whisper_n": whisper_n,
                "count_mismatch": mismatch,
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    n_mismatch = sum(1 for v in results.values() if v["count_mismatch"])
    print(f"\n{n_mismatch} / {len(results)} clips have a Whisper/declared "
          f"word-count mismatch -> {OUT.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
