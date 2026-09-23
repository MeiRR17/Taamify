"""
Calibrate on audio cut the way the product cuts it.
====================================================
THE GAP THIS CLOSES, AND IT IS THE ONE THAT WOULD HAVE SHIPPED

Every calibration this project has run -- including the careful one that
finally derived thresholds from the context-blended score on a parasha-split
model -- read its spectrograms out of `features_v3_word/BEN13R_X.npy`. Those
were cut once, offline, by the corpus builder.

`/api/check_phrase` does not do that. It receives a recording, runs the CTC
aligner to find each word inside it, and cuts a spectrogram from the raw
waveform at those boundaries. Different boundaries, different spectrogram,
different score.

Measured through the live endpoint on Ben13's own correct readings from
held-out parashot: only **10%** of correct words cleared say_wrong_below,
against the 98% the calibration promised. The model was fine -- 86.2% on
those same parashot. The product would have told a correct reader they erred
nine times out of ten.

Nothing failed. The endpoint answered every request.

SO THE THRESHOLDS ARE DERIVED HERE THROUGH THE ENDPOINT ITSELF

Real clips, posted over HTTP, exactly as the app posts them. The positives
are Ben13's own correct readings. The negatives are the same clips submitted
against the ta'am the model most nearly confuses each word with, which is the
mistake a learner actually makes.

This is slower than reading a .npy, and it is the only measurement that
describes the thing the learner will use.

Usage:
  # with the server running on :8000
  python Data_Prep/calibrate_end_to_end.py --clips 200
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
import taam_grammar as tg          # noqa: E402
from holdout import HELD_OUT       # noqa: E402

# Which ta'am each one is actually mistaken for, measured over the corpus.
# Loaded once; falls back to a neighbour of the same class when absent.
def _load_confusion():
    f = P / "results" / "calibration" / "confusion_partner.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


CONTRADICT_BUDGET = 0.02
BLESS_BUDGET = 0.10
URL = "http://127.0.0.1:8000/api/check_phrase"


def held_out_clips(limit):
    """Ben13's own recordings from the held-out parashot, longest first.

    Longer clips first because the sequence tagger has more to work with and
    because they are closer to what a learner submits -- a whole phrase, not
    a fragment.
    """
    man = json.loads((P / "data" / "ben13_reading" /
                      "manifest_yerushalmi.json").read_text(encoding="utf-8"))
    blocks = defaultdict(list)
    for e in man:
        blocks[e["local_path"]].append(e)
    held = {int(p) for p in HELD_OUT}
    out = []
    for path, rows in blocks.items():
        m = re.search(r"/yerushalmi/(\d+)-", path)
        if not m or int(m.group(1)) not in held:
            continue
        if not (P / path).exists():
            continue
        rows = sorted(rows, key=lambda e: int(e["id"]))
        words = [e["text"] for e in rows]
        taamim = []
        for w in words:
            toks = tg.verse_tokens(w)
            taamim.append(toks[0][1] if toks else None)
        if not all(taamim) or len(words) < 2:
            # A one-word "phrase" is the degenerate case, not the product's
            # case: the sequence tagger has no neighbours to reason over, so
            # at CONTEXT_WEIGHT 0.7 the blend is dominated by a context term
            # computed from nothing. A first run filled its whole quota with
            # these and concluded the system could not separate correct from
            # wrong at all -- an artefact of the input, not a property of the
            # model. The learner always submits several words.
            continue
        out.append((str(P / path), words, taamim))
    out.sort(key=lambda t: -len(t[1]))
    if len(out) < limit:
        out += splice_phrases(man, held, limit - len(out))
    return out[:limit]


def splice_phrases(man, held, want, per_phrase=5):
    """Join consecutive single-word recordings of one verse into a phrase.

    Ben13's corpus is recorded one word per file, so there is almost no
    multi-word audio in it -- but a learner submits a whole phrase, and the
    aligner's behaviour over several words is exactly what needs calibrating.
    Concatenating consecutive words of the same verse, in order, from the
    same reader and the same session is the closest available stand-in.

    It is a stand-in and not the real thing: the joins have no breath and no
    co-articulation across the boundary, so alignment here is if anything
    easier than on a real learner's phrase. Thresholds derived from it are
    therefore optimistic on alignment and honest on the cut geometry, which
    is the part that was wrong.
    """
    import shutil
    import subprocess as sp
    tmp = P / "results" / "calibration" / "spliced"
    tmp.mkdir(parents=True, exist_ok=True)
    by_verse = defaultdict(list)
    for e in man:
        m = re.search(r"/yerushalmi/(\d+)-", e["local_path"])
        if m and int(m.group(1)) in held and (P / e["local_path"]).exists():
            by_verse[(e["book"], e["perek"], e["pasuk"])].append(e)
    out = []
    for key, rows in by_verse.items():
        rows = sorted(rows, key=lambda e: int(e["id"]))
        for a in range(0, len(rows) - per_phrase + 1, per_phrase):
            chunk = rows[a:a + per_phrase]
            words = [e["text"] for e in chunk]
            taamim = []
            for w in words:
                toks = tg.verse_tokens(w)
                taamim.append(toks[0][1] if toks else None)
            if not all(taamim):
                continue
            name = tmp / f"{key[0]}_{key[1]}_{key[2]}_{a}.wav"
            if not name.exists():
                lst = tmp / "list.txt"
                lst.write_text("".join(
                    f"file '{(P / e['local_path']).resolve()}'\n" for e in chunk))
                r = sp.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(lst), "-ar", "16000", "-ac", "1",
                            str(name)], capture_output=True)
                if r.returncode != 0 or not name.exists():
                    continue
            out.append((str(name), words, taamim))
            if len(out) >= want:
                return out
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
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--out", default="calibration_e2e.json")
    args = ap.parse_args()

    global CONFUSION
    CONFUSION = _load_confusion()
    if not CONFUSION:
        print("  אין טבלת בלבול — הרץ תחילה את build_confusion_partner")
        return
    clips = held_out_clips(args.clips)
    print(f"  {len(clips)} מקטעים מפרשות מוחזקות "
          f"(חציון {int(np.median([len(c[1]) for c in clips]))} מילים)\n")

    pos, neg, aligned, total_words = [], [], 0, 0
    for n, (path, words, taamim) in enumerate(clips, 1):
        d = post(path, words, taamim)
        if not d or not d.get("aligned"):
            continue
        aligned += 1
        heard = []
        for w in d.get("words", []):
            if w.get("confidence") is None:
                continue
            total_words += 1
            pos.append(w["confidence"])
            # The negative must NOT be the model's own prediction for this
            # clip. Using `heard` made the negatives the one answer the model
            # was most confident about, so every "wrong" submission scored
            # near 1.0 and the catch rate read 0% -- a property of the test,
            # not of the system. The realistic learner error is the ta'am
            # this one is confused with IN GENERAL, taken from the corpus
            # confusion matrix and independent of this recording.
            heard.append(CONFUSION.get(w.get("expected")))
        # resubmit the same audio against those near-miss ta'amim
        if any(heard) and len(heard) == len(taamim):
            wrong = [h or t for h, t in zip(heard, taamim)]
            dn = post(path, words, wrong)
            for w in (dn or {}).get("words", []):
                if w.get("confidence") is not None:
                    neg.append(w["confidence"])
        if n % 25 == 0:
            print(f"    {n}/{len(clips)}", flush=True)

    pos, neg = np.array(pos), np.array(neg)
    print(f"\n  {aligned}/{len(clips)} מקטעים יושרו · "
          f"{len(pos):,} מילים נכונות · {len(neg):,} שליליות")
    if len(pos) < 100 or len(neg) < 100:
        print("  מעט מדי דגימות לכיול אמין.")
        return

    say_wrong = float(np.percentile(pos, CONTRADICT_BUDGET * 100))
    say_ok = float(np.percentile(neg, (1 - BLESS_BUDGET) * 100))
    band = say_ok > say_wrong
    if not band:
        say_ok = say_wrong

    print(f"\n  say_wrong_below   = {say_wrong:.4f}")
    print(f"  say_correct_above = {say_ok:.4f}")
    print(f"  רצועה ניטרלית: {'קיימת' if band else 'ריקה'}\n")
    print(f"  סותר קריאה נכונה        "
          f"{float((pos < say_wrong).mean())*100:5.1f}%")
    print(f"  מאשר טעות                {float((neg >= say_ok).mean())*100:5.1f}%")
    print(f"  תופס טעות                {float((neg < say_wrong).mean())*100:5.1f}%")

    out = P / "results" / "calibration" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "say_wrong_below": round(say_wrong, 4),
        "say_correct_above": round(say_ok, 4),
        "has_neutral_band": band,
        "contradict_rate": round(float((pos < say_wrong).mean()), 4),
        "bless_rate": round(float((neg >= say_ok).mean()), 4),
        "catch_rate": round(float((neg < say_wrong).mean()), 4),
        "contradict_budget": CONTRADICT_BUDGET,
        "bless_budget": BLESS_BUDGET,
        "n_positive": len(pos), "n_negative": len(neg),
        "n_clips": aligned,
        "held_out_parashot": sorted(HELD_OUT),
        "derived_on": ("real clips posted to /api/check_phrase -- the "
                       "aligner's own cuts, not the corpus's pre-cut windows"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  נשמר: {out}")


if __name__ == "__main__":
    main()
