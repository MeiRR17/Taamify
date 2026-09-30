"""
A full aliyah, read block by block through the real product endpoint.
=======================================================================
Every measurement so far has scored isolated blocks. This is the first
artifact that reads like the product: real consecutive audio from one
held-out parasha, in verse order, each block posted to /api/check_phrase
exactly as the app posts it, with the actual headline and per-word summary
the learner would see.

It is not a single continuous recording -- Ben13's corpus is stored as
per-block files, so this walks the blocks of one parasha in order rather
than concatenating them into one file. Every block IS real continuous
audio (2-8 words, real co-articulation, real per-word boundaries), so this
is not the synthetic splice that was already found and discarded; it is
simply presented block by block instead of as one file.

Usage:
  python Data_Prep/aliyah_walkthrough.py --parasha 5 --limit 30
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
from holdout import NAMES        # noqa: E402
import taam_grammar as tg        # noqa: E402

URL = "http://127.0.0.1:8000/api/check_phrase"


def blocks_for_parasha(parasha):
    meta = json.loads((P / "features_v3_word" / "BEN13R_meta.json")
                      .read_text(encoding="utf-8"))
    man = json.loads((P / "data" / "ben13_reading" /
                      "manifest_yerushalmi.json").read_text(encoding="utf-8"))
    audio = defaultdict(set)
    for e in man:
        audio[(e["book"], e["perek"], e["pasuk"])].add(e["local_path"])
    g = defaultdict(list)
    for i, m in enumerate(meta):
        if str(m.get("parasha")) == str(parasha):
            g[(m["verse_ref"], m["segment_block"])].append(i)
    import librosa
    out = []
    for (ref, _blk), idxs in sorted(g.items()):
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["start"])
        b, c, v = ref.split(".")
        chosen = None
        for pth in audio.get((b, c, v), ()):
            if not (P / pth).exists():
                continue
            try:
                d = librosa.get_duration(path=str(P / pth))
            except Exception:
                continue
            if abs(d - meta[idxs[-1]]["end"]) < 0.4:
                chosen = str(P / pth)
                break
        if not chosen:
            continue
        out.append((ref, chosen, [meta[i]["word_raw"] for i in idxs],
                    [meta[i]["taam"] for i in idxs]))
    return out


def post(path, words, taamim):
    r = subprocess.run(
        ["curl", "-s", "-m", "120", "-X", "POST", URL,
         "-F", f"audio=@{path}",
         "-F", "words=" + json.dumps(words, ensure_ascii=False),
         "-F", "taamim=" + json.dumps(taamim, ensure_ascii=False)],
        capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parasha", default="5")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    blocks = blocks_for_parasha(args.parasha)[:args.limit]
    name = NAMES.get(str(args.parasha), args.parasha)
    print(f"  פרשת {name} — {len(blocks)} מקטעים\n")
    print("=" * 70)

    n_ok = n_repeat = n_note = n_gated = 0
    for ref, path, words, taamim in blocks:
        d = post(path, words, taamim)
        text = " ".join(words)
        if not d or not d.get("aligned"):
            n_gated += 1
            print(f"\n  {ref}   {text}")
            print(f"    ⚪ {(d or {}).get('message', 'שגיאה')}")
            continue
        rep = d.get("report") or {}
        head = rep.get("headline", "?")
        if rep.get("should_repeat"):
            n_repeat += 1
            mark = "🔴"
        elif "לא הצלחתי" in head:
            n_note += 1
            mark = "⚪"
        elif head == "הקטע נקרא נכון":
            n_ok += 1
            mark = "🟢"
        else:
            n_note += 1
            mark = "🟡"
        print(f"\n  {ref}   {text}")
        print(f"    {mark} {head}")
        for w in d.get("words", []):
            if w.get("summary"):
                print(f"       · {w['word']}: {w['summary']}")

    print("\n" + "=" * 70)
    tot = n_ok + n_repeat + n_note + n_gated
    print(f"\n  {tot} מקטעים: 🟢 נכון {n_ok} · 🟡 הערה {n_note} · "
          f"🔴 חזרה {n_repeat} · ⚪ לא ברור/נדחה {n_gated}")


if __name__ == "__main__":
    main()
