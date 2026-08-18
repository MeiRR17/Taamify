"""
How wrong are Whisper's word boundaries on chanted Torah?
=========================================================
R1/R2's word windows come from Whisper; R6's come from PocketTorah's
human-verified onsets. Listening to R1/R2 clips, words bleed into their
neighbours and get cut mid-syllable — so the windows need a number, not a
proxy. The CTC success rate reported earlier is not that number: it only says
the word lies *somewhere* inside the given window.

R6 supplies the missing ground truth. Running the exact same transcription
settings as transcribe_mac.py over PocketTorah audio, aligning the result the
same way, and differencing against the gold onsets measures Whisper's word
boundary error on this material directly — chanted, melismatic Hebrew, which is
nothing like the speech Whisper was trained on.

The output is the error distribution of start and end times, in milliseconds.
That decides whether R1/R2's 51,191 words need re-alignment from the acoustics.

Usage:
  python Data_Prep/whisper_vs_gold.py --aliyot 3
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import normalize_word, skeleton  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_pt.json"
CACHE = Path.home() / ".cache" / "taamify" / "whisper_pt"
MODEL = "mlx-community/whisper-large-v3-mlx"      # identical to transcribe_mac.py


def transcribe(audio_path: Path) -> list:
    """Flat word list from Whisper, cached per recording."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / (audio_path.stem + ".json")
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    try:
        import truststore
        truststore.inject_into_ssl()      # TLS through the inspecting proxy
    except ImportError:
        pass
    import mlx_whisper
    t0 = time.time()
    result = mlx_whisper.transcribe(
        str(audio_path), path_or_hf_repo=MODEL,
        language="he", word_timestamps=True, verbose=False,
    )
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            norm = normalize_word(w.get("word", ""))
            if len(norm) < 2:
                continue
            words.append({"norm": norm, "start": float(w["start"]),
                          "end": float(w["end"]),
                          "probability": float(w.get("probability", 0.0))})
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    print(f"    transcribed in {time.time() - t0:.0f}s: {len(words)} words", flush=True)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliyot", type=int, default=3, help="how many recordings to test")
    args = ap.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        gold_records = json.load(f)

    by_file = {}
    for r in gold_records:
        by_file.setdefault(r["audio_file"], []).append(r)

    files = sorted(by_file)[: args.aliyot]
    d_start, d_end, d_dur = [], [], []
    matched = total = 0

    for audio_file in files:
        path = PROJECT_DIR / audio_file
        if not path.exists():
            continue
        gold = sorted(by_file[audio_file], key=lambda r: r["start"])
        print(f"  {path.name}: {len(gold)} gold words", flush=True)
        hyp = transcribe(path)
        if not hyp:
            continue
        total += len(gold)

        # Same positional, skeleton-based alignment clean_align_v2 uses, so any
        # error measured here is Whisper's timing, not a different matcher.
        a = [skeleton(w["norm"]) for w in hyp]
        b = [skeleton(r["word_clean"]) for r in gold]
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for k in range(i2 - i1):
                h, g = hyp[i1 + k], gold[j1 + k]
                d_start.append((h["start"] - g["start"]) * 1000)
                d_end.append((h["end"] - g["end"]) * 1000)
                d_dur.append(((h["end"] - h["start"]) - (g["end"] - g["start"])) * 1000)
                matched += 1

    if not matched:
        print("no words matched — cannot measure")
        return

    print(f"\nmatched {matched:,}/{total:,} gold words "
          f"({100 * matched / max(total, 1):.1f}%)\n")
    print(f"{'':<12}{'median':>10}{'mean':>10}{'|err| p50':>11}"
          f"{'p90':>9}{'p99':>9}{'>200ms':>9}")
    print("-" * 70)
    for name, arr in (("start", d_start), ("end", d_end), ("duration", d_dur)):
        a = np.array(arr)
        print(f"{name:<12}{np.median(a):>9.0f}m{np.mean(a):>9.0f}m"
              f"{np.median(np.abs(a)):>10.0f}m{np.percentile(np.abs(a), 90):>8.0f}m"
              f"{np.percentile(np.abs(a), 99):>8.0f}m"
              f"{100 * np.mean(np.abs(a) > 200):>8.1f}%")
    print("-" * 70)

    med_word = np.median([r["end"] - r["start"] for r in gold_records[:20000]])
    err = np.median(np.abs(d_start))
    print(f"median gold word: {med_word * 1000:.0f} ms")
    print(f"median |start error| is {100 * err / (med_word * 1000):.0f}% of a word.")
    print("A stressed syllable is ~45% of a word, so an error of that size moves\n"
          "the window off the syllable it is supposed to be centred on.")


if __name__ == "__main__":
    main()
