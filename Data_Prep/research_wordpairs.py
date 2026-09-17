"""
The decisive contrast: the SAME WORD sung with the SAME vs a DIFFERENT ta'am.
============================================================================
research_variance.py produced an awkward, and important, result: two readers
singing the same word are CLOSER to each other (D_cross) than one reader is to
himself on two different words of the same ta'am (D_self). Whatever the
contour encodes, the identity of the WORD moves it more than the identity of
the READER does. That makes the naive "portability" ratio uninterpretable and
forces a better-controlled comparison.

This script builds it. Many Hebrew words carry different ta'amim in different
verses -- the same observation build_stage_b.py rests on. So for one fixed word
we can ask:

    same word, same ta'am    vs    same word, different ta'am

with the text, syllable count and stress position identical on both sides of
the comparison. Computed twice:

    within-reader  one voice, two verses.       -> the single-voice ceiling
    cross-reader   two voices, two verses.      -> what a product would face

The gap between them is the honest portability number: how much of a ta'am's
melodic identity survives changing the singer.

Stage 1 (this script) extracts the curve bank; Stage 2 (research_wordtest.py)
runs the contrast.

Usage:
  venv/bin/python Data_Prep/research_wordpairs.py --per-cell 14
"""
import argparse
import json
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from research_extract_curves import track  # noqa: E402
from research_build_index import load_ben13, load_merged  # noqa: E402

OUT = P / "results" / "melody_research" / "curves_wordpairs.json"
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "3"))
SEED = 1


def do_one(r):
    tr = track(r["audio_file"], r["start"], r["end"])
    if tr is None:
        return None
    return {**tr, "reader": r["reader"], "taam": r["taam"],
            "word_clean": r["word_clean"], "ref": r["ref"],
            "word_raw": r.get("word_raw")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=14,
                    help="instances per (word, ta'am) cell")
    ap.add_argument("--min-readers", type=int, default=2)
    args = ap.parse_args()

    rows = load_ben13() + load_merged()
    # (word, ta'am) cells
    cells = defaultdict(list)
    for r in rows:
        cells[(r["word_clean"], r["taam"])].append(r)

    # keep words that appear with >=2 ta'amim, each cell reasonably populated
    by_word = defaultdict(list)
    for (w, t), rs in cells.items():
        readers = {x["reader"] for x in rs}
        if len(rs) >= 8 and len(readers) >= args.min_readers:
            by_word[w].append((t, rs))
    words = {w: v for w, v in by_word.items() if len(v) >= 2}
    print(f"{len(words):,} words carry >=2 ta'amim with >=8 instances each")

    rng = random.Random(SEED)
    picked = []
    for w, lst in words.items():
        for t, rs in lst:
            # spread the sample across readers so a cell is not one voice
            by_r = defaultdict(list)
            for x in rs:
                by_r[x["reader"]].append(x)
            for v in by_r.values():
                rng.shuffle(v)
            order, k = [], 0
            while len(order) < args.per_cell:
                added = False
                for rd in sorted(by_r):
                    if k < len(by_r[rd]):
                        order.append(by_r[rd][k])
                        added = True
                        if len(order) >= args.per_cell:
                            break
                if not added:
                    break
                k += 1
            picked.extend(order)
    print(f"{len(picked):,} curves to extract, {WORKERS} workers")
    print("  by reader " + str(Counter(p["reader"] for p in picked).most_common()))

    out, done = [], 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(do_one, picked, chunksize=32):
            done += 1
            if res is not None:
                out.append(res)
            if done % 2000 == 0:
                print(f"  {done}/{len(picked)} kept={len(out):,}", flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(out):,} curves -> {OUT.relative_to(P)}")


if __name__ == "__main__":
    main()
