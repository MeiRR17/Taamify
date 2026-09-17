"""
Is Ben13's "different" version of a ta'am really his window?
============================================================
Ben13's median word is 5.8 s for Zaqef Gadol and 4.3 s for Gershayim against
1.5-1.7 s for R1/R2/R8, and those are exactly the ta'amim where his contour
features look nothing like anyone else's. His corpus is cut into phrases at the
disjunctive accents, so a disjunctive word is the last word before a cut.

Direct test: re-extract the SAME Ben13 words with the window truncated to the
median duration the other three readers give that ta'am, and see whether his
feature means move toward theirs. If they do, the disagreement is a window
artifact and no melodic claim should be made from it. If they do not, Ben13
really is singing something else.

Usage:
  venv/bin/python Data_Prep/research_truncate.py
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from research_extract_curves import track  # noqa: E402
from research_melody_lib import features  # noqa: E402

MR = P / "results" / "melody_research"
OUT = MR / "truncate.json"
WORKERS = int(os.environ.get("TAAMIFY_WORKERS", "3"))
TARGETS = ["Zaqef_Gadol", "Gershayim", "Geresh", "Telisha_Qetana",
           "Telisha_Gedola", "Yetiv", "Pazer", "Tevir", "Revia", "Siluk"]
PER = 90
SHOW = ["end_vs_tonic", "head_vs_tonic", "turns", "range", "net", "trough_pos"]


def job(a):
    path, s, e = a
    return track(path, s, e)


def main():
    idx = json.loads((MR / "matched_index.json").read_text(encoding="utf-8"))
    rows = json.loads((MR / "curves_matched.json").read_text(encoding="utf-8"))

    other_med = {}
    d = defaultdict(list)
    for r in rows:
        if r["reader"] != "BEN13R":
            d[r["taam"]].append(r["dur"])
    for t, v in d.items():
        other_med[t] = float(np.median(v))

    rng = np.random.default_rng(0)
    work, meta = [], []
    for t in TARGETS:
        cand = [s for s in idx if s["taam"] == t and "BEN13R" in s["readers"]]
        if not cand:
            continue
        rng.shuffle(cand)
        for s in cand[:PER]:
            r = s["readers"]["BEN13R"]
            cap = min(r["end"], r["start"] + other_med.get(t, 1.2))
            if cap - r["start"] < 0.3:
                continue
            work.append((r["audio_file"], r["start"], cap))
            meta.append({"taam": t, "stress": s["stress"],
                         "ref": s["ref"], "dur_full": r["end"] - r["start"],
                         "dur_cut": cap - r["start"]})
    print(f"{len(work)} Ben13 words re-extracted with truncated windows, "
          f"{WORKERS} workers")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        res = list(pool.map(job, work, chunksize=16))

    # tonic per book from the FULL extraction, so the reference pitch is
    # unchanged between the two conditions
    by_rb = defaultdict(list)
    for r in rows:
        by_rb[(r["reader"], r["ref"].rsplit(" ", 1)[0])].append(r["med_hz"])
    tonic = {k: float(np.median(v)) for k, v in by_rb.items()}

    cut = defaultdict(list)
    for m, tr in zip(meta, res):
        if tr is None:
            continue
        book = m["ref"].rsplit(" ", 1)[0]
        off = 12 * np.log2(tr["med_hz"] / tonic[("BEN13R", book)])
        cut[m["taam"]].append(features(tr["curve"], tr["dur"], m["stress"], off))

    full, others = defaultdict(list), defaultdict(list)
    for r in rows:
        book = r["ref"].rsplit(" ", 1)[0]
        off = 12 * np.log2(r["med_hz"] / tonic[(r["reader"], book)])
        f = features(r["curve"], r["dur"], r.get("stress"), off)
        (full if r["reader"] == "BEN13R" else others)[r["taam"]].append(f)

    out = {}
    print(f"\n{'taam':<17}{'feature':<15}{'B13 full':>10}{'B13 cut':>10}"
          f"{'others':>9}{'gap closed':>12}")
    for t in TARGETS:
        if t not in cut or len(cut[t]) < 20:
            continue
        out[t] = {"n_cut": len(cut[t])}
        for name in SHOW:
            a = np.nanmedian([f[name] for f in full[t]])
            b = np.nanmedian([f[name] for f in cut[t]])
            c = np.nanmedian([f[name] for f in others[t]])
            gap = a - c
            closed = (a - b) / gap if abs(gap) > 1e-6 else float("nan")
            out[t][name] = {"b13_full": float(a), "b13_cut": float(b),
                            "others": float(c), "gap_closed": float(closed)}
            print(f"{t:<17}{name:<15}{a:>10.2f}{b:>10.2f}{c:>9.2f}{closed:>12.2f}")
        print()

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT.relative_to(P)}")


if __name__ == "__main__":
    main()
