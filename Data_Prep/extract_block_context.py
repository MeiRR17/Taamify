"""
Real continuous reading, already on disk.
==========================================
The corpus is not the teaching corpus I took it for. Grouping BEN13R by
(verse_ref, segment_block) gives 21,587 blocks of 2-8 words -- real
continuous audio, with real co-articulation and breath, and per-word
boundaries the aligner already computed. Median word there is 1.04s, against
3.30s in the single-word blocks and 0.4-0.6s in a child's reading. The gap is
2x, not the 8x I measured off the single-word subset.

This marks every corpus row with the size of its block and its duration, so
training and evaluation can be restricted to, or weighted toward, words that
were actually read in sequence.

Usage:
  python Data_Prep/extract_block_context.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
F = P / "features_v3_word"


def main():
    meta = json.loads((F / "BEN13R_meta.json").read_text(encoding="utf-8"))
    blocks = defaultdict(list)
    for i, m in enumerate(meta):
        blocks[(m["verse_ref"], m["segment_block"])].append(i)

    for key, idxs in blocks.items():
        n = len(idxs)
        ordered = sorted(idxs, key=lambda i: meta[i]["start"])
        for pos, i in enumerate(ordered):
            meta[i]["block_size"] = n
            meta[i]["block_pos"] = pos
            meta[i]["duration"] = round(meta[i]["end"] - meta[i]["start"], 3)
            meta[i]["read_in_sequence"] = n > 1

    (F / "BEN13R_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    d = np.array([m["duration"] for m in meta if m["read_in_sequence"]])
    s = np.array([m["duration"] for m in meta if not m["read_in_sequence"]])
    seq = sum(1 for m in meta if m["read_in_sequence"])
    print(f"  {len(blocks):,} בלוקים · {seq:,} מילים נקראו ברצף "
          f"({seq/len(meta)*100:.0f}%)")
    print(f"  משך: ברצף חציון {np.median(d):.2f}s · בודדות {np.median(s):.2f}s")
    for lo, hi, name in ((0, 0.8, "מהיר"), (0.8, 1.5, "בינוני"), (1.5, 99, "איטי")):
        n = int(((d >= lo) & (d < hi)).sum())
        print(f"    {name:<8}{n:>7,}  ({n/len(d)*100:4.0f}%)")


if __name__ == "__main__":
    main()
