"""
Build continuous reading audio out of the teaching corpus.
===========================================================
THE PROBLEM THIS SOLVES

Ben13's corpus is one word per file, chanted deliberately over a 4.2s median.
A child reading continuously produces words in 0.4-0.6s. Measured on held-out
words, the model scores 94.4% at the corpus's tempo, 26.4% at 2x and 0.0% at
4x -- it has never heard a ta'am at reading tempo, and no window geometry
recovers an 8x mismatch.

The corpus does, however, know which word follows which: every row carries a
verse reference and an index. So continuous reading can be constructed rather
than recorded.

HOW

For each verse, take its words in order, compress each one to reading tempo,
and butt them together with a short crossfade and no silence between. The
result is one waveform per verse at a realistic tempo, and -- because it was
assembled here -- the exact boundary of every word inside it is known, which
no forced aligner can promise.

Two things this gives that the recorded corpus cannot:

  training data in the target domain -- words at reading tempo, in context,
  with their true labels

  an evaluation set with ground-truth boundaries, so the aligner's errors and
  the model's errors can finally be told apart

WHAT IT IS NOT

It is a reconstruction, not a recording. Real continuous reading has
co-articulation across word boundaries, breath groups, and a tempo that
varies with the ta'amim -- a word carrying Etnachta is genuinely drawn out.
None of that is here. The crossfade is a seam, not a transition.

So a model trained on this has learned to hear ta'amim at reading tempo,
which it currently cannot do at all, but it has NOT been shown to work on a
real child. That still needs one recording of someone reading continuously,
and this does not substitute for it -- it makes the gap smaller and testable.

TEMPO

Each word is compressed by a factor drawn per verse, so a verse has a
consistent tempo, with word-level jitter on top -- a reader has one pace and
wobbles within it, rather than re-choosing speed every word.

Usage:
  python Data_Prep/build_continuous_reading.py --verses 400
  python Data_Prep/build_continuous_reading.py --verses 2000 --out-dir data/ben13_continuous
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
import taam_grammar as tg          # noqa: E402

SR = 16000
CROSSFADE_S = 0.02          # a seam short enough not to smear the melisma
TARGET_WORD_S = (0.42, 0.62)   # what a word takes in continuous reading


def parasha_of(path):
    """Parasha number from the folder name, e.g. .../yerushalmi/05-chayei/..."""
    m = re.search(r"/yerushalmi/(\d+)-", path)
    return m.group(1).lstrip("0") or "0" if m else None


def compress(y, target_s, rng):
    """Squeeze a word to the target duration. librosa's phase vocoder is used
    rather than plain resampling, so the pitch -- which is the ta'am -- is
    preserved while the tempo changes."""
    import librosa
    cur = len(y) / SR
    if cur <= 0.05:
        return None
    rate = cur / max(target_s, 0.05)
    if rate <= 1.01:
        return y
    try:
        return librosa.effects.time_stretch(y=y, rate=float(rate))
    except Exception:
        return None


def join(pieces):
    """Butt the words together with a short crossfade and no silence."""
    n_fade = int(CROSSFADE_S * SR)
    out = pieces[0]
    bounds = [(0.0, len(out) / SR)]
    for p in pieces[1:]:
        f = min(n_fade, len(out), len(p))
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            tail = out[-f:] * (1 - ramp) + p[:f] * ramp
            start = (len(out) - f) / SR
            out = np.concatenate([out[:-f], tail, p[f:]])
        else:
            start = len(out) / SR
            out = np.concatenate([out, p])
        bounds.append((start, len(out) / SR))
    return out, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", type=int, default=400)
    ap.add_argument("--out-dir", default="data/ben13_continuous")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    rng = np.random.default_rng(args.seed)
    man = json.loads((P / "data" / "ben13_reading" /
                      "manifest_yerushalmi.json").read_text(encoding="utf-8"))

    # group by verse, in reading order
    verses = defaultdict(list)
    for e in man:
        verses[(e["book"], e["perek"], e["pasuk"])].append(e)

    out_dir = P / args.out_dir
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    keys = sorted(verses, key=lambda k: tuple(int(x) for x in k))
    built, index = 0, []
    for key in keys:
        if built >= args.verses:
            break
        rows = sorted(verses[key], key=lambda e: int(e["id"]))
        if len(rows) < args.min_words:
            continue
        # one pace per verse, jitter per word
        base = float(rng.uniform(*TARGET_WORD_S))
        pieces, words, taamim, ok = [], [], [], True
        for e in rows:
            path = P / e["local_path"]
            if not path.exists():
                ok = False
                break
            toks = tg.verse_tokens(e["text"])
            if not toks or not toks[0][1]:
                ok = False
                break
            try:
                y, _ = librosa.load(str(path), sr=SR)
            except Exception:
                ok = False
                break
            target = float(np.clip(base * rng.uniform(0.85, 1.15), 0.3, 1.2))
            c = compress(y, target, rng)
            if c is None or len(c) < int(0.1 * SR):
                ok = False
                break
            pieces.append(c.astype(np.float32))
            words.append(e["text"])
            taamim.append(toks[0][1])
        if not ok or len(pieces) < args.min_words:
            continue

        audio, bounds = join(pieces)
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak * 0.9
        ref = f'{key[0]}.{key[1]}.{key[2]}'
        name = f"{ref}.wav"
        sf.write(str(out_dir / "audio" / name), audio, SR)
        index.append({
            "ref": ref, "file": f"audio/{name}",
            "parasha": parasha_of(rows[0]["local_path"]),
            "duration": round(len(audio) / SR, 3),
            "words": words, "taamim": taamim,
            "bounds": [[round(a, 3), round(b, 3)] for a, b in bounds],
            "mean_word_s": round(float(np.mean([b - a for a, b in bounds])), 3),
        })
        built += 1
        if built % 50 == 0:
            print(f"    {built}/{args.verses}", flush=True)

    (out_dir / "index.json").write_text(
        json.dumps({
            "n_verses": len(index),
            "n_words": sum(len(v["words"]) for v in index),
            "sr": SR,
            "crossfade_s": CROSSFADE_S,
            "target_word_s": list(TARGET_WORD_S),
            "note": ("reconstructed continuous reading: each word from the "
                     "teaching corpus compressed to reading tempo and joined "
                     "with a short crossfade. Boundaries are exact because "
                     "the audio was assembled here. Not a recording -- no "
                     "co-articulation, no breath groups, no ta'am-driven "
                     "tempo variation."),
            "verses": index,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    if index:
        w = np.array([v["mean_word_s"] for v in index])
        print(f"\n  {len(index)} פסוקים · "
              f"{sum(len(v['words']) for v in index):,} מילים")
        print(f"  משך מילה: חציון {np.median(w):.2f}s "
              f"(קורפוס המקור: 4.20s)")
        print(f"  נשמר: {out_dir}")
    else:
        print("  לא נבנה כלום.")


if __name__ == "__main__":
    main()
