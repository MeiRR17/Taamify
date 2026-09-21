"""
A real error test set, built from correct readings only.
=========================================================
THE PROBLEM

Stage B — telling a learner WHICH mistake they made — has never been
measured, because every reader in the corpus reads correctly. There is not
one example of a wrong reading to test against, and the plan said this was
blocked until real learners recorded mistakes.

It is not blocked.

THE OBSERVATION

A Hebrew word appears many times across the Torah, and not always with the
same ta'am. וַיֹּאמֶר carries Munach 115 times and Mahapakh 114 times;
אֲשֶׁר carries Mercha 327 times and Munach 261. Measured over this corpus:
**5,021 distinct words appear with two or more different ta'amim**, across
56,522 audio rows, yielding 48,356 (word, wrong-ta'am) combinations.

So the audio of וַיֹּאמֶר sung with Munach already exists, and so does the
audio of the same word sung with Mahapakh. Present the first where the text
demands Mahapakh and you have a **genuine recording of that exact word sung
with the wrong melody** — not a synthetic splice, not a pitch-shifted fake,
but a real ba'al kri'ah really singing the wrong ta'am for that position.

That is precisely the error a learner makes.

WHAT THIS DOES AND DOES NOT COVER

Covers: the wrong-ta'am error, which is the dominant failure mode and the
one the product's core promise is about.

Does not cover: hesitation, restart, a skipped word, or a word read with the
right ta'am but wrong stress or pronunciation. Those need the other channels
or real learner audio, and are NOT claimed here.

One honest caveat: the word is real and the melody is real, but they come
from a different verse, so the surrounding context differs. For a per-word
verifier — which is what Stage A is — that is exactly the right test. For a
verse-level sequence model it would be a weaker one, so this set is used to
evaluate the word-level verdict only.

Usage:
  python Data_Prep/build_stage_b.py            # build + evaluate
  python Data_Prep/build_stage_b.py --n 2000
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
FEAT = P / "features_v3_word"
OUT = P / "results" / "stage_b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500,
                    help="how many error cases to build")
    ap.add_argument("--reader", default="BEN13R")
    args = ap.parse_args()

    X = np.load(FEAT / f"{args.reader}_X.npy", mmap_mode="r")
    meta = json.loads((FEAT / f"{args.reader}_meta.json").read_text(encoding="utf-8"))

    spec_mod = importlib.util.spec_from_file_location("srv", P / "webapp" / "server.py")
    srv = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(srv)
    st = srv.get_model()
    classes = set(st["class_names"])

    # hold out the same parashot the honest split holds out, so nothing here
    # was seen in training
    rng = np.random.default_rng(42)
    sys.path.insert(0, str(P / "Data_Prep"))
    from holdout import HELD_OUT
    held = set(HELD_OUT)

    by_word = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        if str(m.get("parasha")) in held and m["taam"] in classes:
            by_word[m["word_clean"]][m["taam"]].append(i)

    multi = {w: d for w, d in by_word.items() if len(d) >= 2}
    print(f"held-out parashot: {sorted(held)}")
    print(f"{len(multi):,} words there carry more than one ta'am\n")

    cases = []
    words = sorted(multi)
    rng.shuffle(words)
    for w in words:
        d = multi[w]
        taamim = sorted(d)
        for a in taamim:
            for b in taamim:
                if a == b:
                    continue
                # audio really sung with `a`, presented where the text wants `b`
                cases.append({"word": w, "row": int(rng.choice(d[a])),
                              "sung": a, "text_requires": b})
        if len(cases) >= args.n * 3:
            break
    rng.shuffle(cases)
    cases = cases[:args.n]
    print(f"{len(cases):,} real wrong-reading cases built\n")

    # how does the shipped system do on them?
    caught = missed = 0
    named_right = 0
    conf_wrong, conf_right = [], []
    by_pair = Counter()
    for c in cases:
        spec = np.array(X[c["row"]])
        # the same decision function the endpoint runs, per-class thresholds
        # included -- otherwise this measures a system nobody is served
        prob, _ = srv.score_word(spec, c["text_requires"])
        if srv.decide(prob, c["text_requires"]) == "wrong":
            caught += 1
            conf_wrong.append(prob)
            top3, _ = srv.classify(spec)
            if top3 and top3[0][0] == c["sung"]:
                named_right += 1
        else:
            missed += 1
            conf_right.append(prob)
            by_pair[(c["text_requires"], c["sung"])] += 1

    # and the control: the SAME audio against the ta'am it really is
    ctrl_ok = 0
    for c in cases[:500]:
        prob, _ = srv.score_word(np.array(X[c["row"]]), c["sung"])
        ctrl_ok += 1 if srv.decide(prob, c["sung"]) == "ok" else 0

    L = []
    def say(s=""):
        L.append(s); print(s)

    say("=" * 66)
    say("  Stage B on REAL wrong readings")
    say("=" * 66)
    say(f"\n  {len(cases):,} cases: a word genuinely sung with ta'am A,")
    say(f"  presented where the text requires ta'am B\n")
    say(f"  caught as wrong        {caught:>5}  {caught/len(cases)*100:5.1f}%")
    say(f"  missed (called fine)   {missed:>5}  {missed/len(cases)*100:5.1f}%")
    if caught:
        say(f"\n  of the caught ones, the system named the ACTUAL ta'am sung")
        say(f"  correctly in {named_right}/{caught} = {named_right/caught*100:.1f}%")
        say(f"  -- that is the Stage B question: not 'wrong' but 'you sang X'")
    say(f"\n  control: the same audio against the ta'am it really is,")
    say(f"  confirmed {ctrl_ok}/500 = {ctrl_ok/500*100:.1f}%")
    if by_pair:
        say(f"\n  hardest confusions (text wanted -> actually sung):")
        for (want, sung), n in by_pair.most_common(8):
            say(f"    {want:<16} -> {sung:<16} missed {n}")
    say("\n" + "=" * 66)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "cases.json").write_text(json.dumps(cases[:200], ensure_ascii=False,
                                               indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({
        "n_cases": len(cases), "caught": caught, "missed": missed,
        "catch_rate": round(caught / len(cases), 4),
        "named_actual_taam": round(named_right / caught, 4) if caught else None,
        "control_confirm_rate": round(ctrl_ok / 500, 4),
        "held_out_parashot": sorted(held),
        "source": "same word sung with a different ta'am elsewhere in the corpus",
    }, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
