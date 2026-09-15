"""
Find words Ben13's text declares but its audio never sings.
===========================================================
The core defect found by the manual review: Ben13's declared text
sometimes lists a ta'am name that isn't actually sung in the clip. A
forced alignment handed too many words can't fail loudly -- it
compresses monotonically, so every boundary after the phantom word
shifts and every segment past it is mislabeled, with an unremarkable
confidence score. Nothing downstream catches it.

METHOD (leave-one-out CTC hypothesis testing)
---------------------------------------------
CTC's alignment score is a likelihood. So: align the declared word list,
then align every leave-one-out variant, and see which fits the audio
better. Forcing a phantom token onto audio that doesn't contain it costs
real likelihood, so removing it produces a large score gain; removing a
word that IS present costs information and produces little or none.

WHY THE OBVIOUS ALTERNATIVES WERE REJECTED (all tried, all measured on
the same 13-clip hand-labelled ground truth taken from the reviewer's
own notes):

  silence-gap counting     3/13  -- these clips have no silence between
                                    words; the teacher sings continuously
  onset detection          3/13  -- melisma produces onsets mid-word
  spectral novelty (SSM)   0/13  -- no usable structure boundaries
  F0 unvoiced-gap counting 6/13  (8/13 after tuning) -- best of the
                                    audio-only family, still not usable
  per-word CTC score       4/12  -- a phantom word doesn't reliably score
                                    worst; the damage spreads to neighbours
  leave-one-out CTC        7/7   -- on every decidable clip

CALIBRATION AND ITS LIMIT
-------------------------
Dropping a word always improves the score somewhat, and the shorter the
list the bigger that free gain -- on a 2-word clip, dropping one leaves a
single word CTC can place almost anywhere, so the test is meaningless
there. Hence MIN_TOKENS=3, and clips below it are reported UNDECIDABLE
rather than guessed at. GAIN_THRESHOLD=0.90 separated true phantoms
(1.09-1.68) from the largest false gain on a known-good clip (0.50) with
a clear margin on the labelled set.

This finds a MISSING-word defect. It does not detect a wrong word, a
wrong order, or a wrong ta'am label -- those need an ear.

Usage:
  python Data_Prep/detect_phantom_words.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import normalize_word            # noqa: E402
from ctc_align import load_recording                 # noqa: E402
from verse_align import align_verse                  # noqa: E402
from segment_ben13_clips import clean_text           # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEN13_DIR = PROJECT_DIR / "data" / "ben13"
OUT = PROJECT_DIR / "data" / "ben13_segments" / "phantom_words.json"

MIN_TOKENS = 3        # below this the leave-one-out test is meaningless
GAIN_THRESHOLD = 0.90  # calibrated on 13 hand-labelled clips, see docstring


def score(y, tokens) -> float:
    spans, sc, _ = align_verse(
        y, [{"raw": t, "norm": normalize_word(t)} for t in tokens])
    return sc if spans is not None else -99.0


def main():
    results = {}
    for type_name in ["teamimtora", "teamim-meg", "teamim-haf"]:
        mpath = BEN13_DIR / type_name / "manifest.json"
        if not mpath.exists():
            continue
        with open(mpath, encoding="utf-8") as f:
            entries = json.load(f)
        combos = [e for e in entries if len(e["taam_sequence"]) > 1]
        print(f"\n{'=' * 78}\n  {type_name}\n{'=' * 78}")
        for e in sorted(combos, key=lambda x: x["order"]):
            path = PROJECT_DIR / e["local_path"]
            if not path.exists():
                continue
            key = f"{type_name}-{e['order']}"
            tokens = clean_text(e.get("textSfaradi") or e.get("text") or "").split()
            if len(tokens) < MIN_TOKENS:
                results[key] = {"verdict": "undecidable", "n_tokens": len(tokens),
                                "reason": f"fewer than {MIN_TOKENS} words"}
                print(f"  order {e['order']:>3}  UNDECIDABLE ({len(tokens)} words)")
                continue

            y = load_recording(path)
            full = score(y, tokens)
            drops = [(score(y, tokens[:i] + tokens[i + 1:]), i)
                     for i in range(len(tokens))]
            best_score, best_i = max(drops)
            gain = best_score - full
            phantom = gain > GAIN_THRESHOLD

            results[key] = {
                "verdict": "phantom_word" if phantom else "ok",
                "n_tokens": len(tokens), "tokens": tokens,
                "full_score": round(full, 3),
                "best_drop_index": best_i, "best_drop_score": round(best_score, 3),
                "gain": round(gain, 3),
                "phantom_index": best_i if phantom else None,
                "phantom_word": tokens[best_i] if phantom else None,
            }
            mark = f"  <- PHANTOM: '{tokens[best_i]}' (#{best_i})" if phantom else ""
            print(f"  order {e['order']:>3}  full {full:>6.2f}  "
                  f"best-drop #{best_i} {best_score:>6.2f}  gain {gain:>5.2f}{mark}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    n_ph = sum(1 for v in results.values() if v["verdict"] == "phantom_word")
    n_un = sum(1 for v in results.values() if v["verdict"] == "undecidable")
    print(f"\n{'=' * 78}")
    print(f"  {n_ph} clips with a detected phantom word")
    print(f"  {n_un} undecidable (too few words to test)")
    print(f"  {len(results) - n_ph - n_un} clips look consistent")
    print(f"-> {OUT.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
