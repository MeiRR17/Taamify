"""
Does judging a word in its phrase beat judging it alone? -- measured.
=====================================================================
verify_match() scores one word against one expected ta'am with no idea what
surrounds it. The sequence tagger knows the whole phrase and is worth +6.8
points on classification, but until now it reached only /api/analyze -- never
the verification decision the product actually runs.

context_probabilities() + combine_verdict() in webapp/server.py now blend the
two. This script asks whether that blend is actually better, and at what
weight, against a fixed error budget: how many real mistakes are caught while
contradicting at most 2% of correct readings. Comparing at a FIXED budget is
the only honest comparison -- a blend that catches more errors by also
contradicting more readers has not improved anything.

RECOVERING WORD ORDER

The corpus metadata is shuffled and carries no word index, so a verse cannot
be reassembled from it directly -- and order is the entire point here. Each
row does carry verse_ref and word_raw, so the order is recovered by matching
the rows back onto the pointed text of that verse. Rows whose word does not
match, or verses not fully covered, are skipped rather than guessed at.

Usage:
  python Data_Prep/eval_context_verify.py
  python Data_Prep/eval_context_verify.py --verses 400
"""
import argparse
import importlib.util
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
import taam_grammar as tg          # noqa: E402

BOOKS = {1: "Genesis", 2: "Exodus", 3: "Leviticus",
         4: "Numbers", 5: "Deuteronomy"}
CONTRADICT_BUDGET = 0.02


def load_server():
    spec = importlib.util.spec_from_file_location("srv", P / "webapp" / "server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


def verse_text(ref):
    """The pointed words of one verse, in order, from the cache."""
    try:
        b, c, v = (int(x) for x in ref.split("."))
    except Exception:
        return None
    path = P / "Data_Prep" / "sefaria_cache" / f"{BOOKS.get(b)}.{c}.json"
    if not path.exists():
        return None
    verses = json.loads(path.read_text(encoding="utf-8"))
    if not 0 < v <= len(verses):
        return None
    # NFC on both sides: the cache and the corpus metadata disagree on
    # composition order for the same pointed word, so an untouched string
    # compare fails on words that are in fact identical. That alone dropped
    # the reconstruction rate to well under 1% on the first run.
    return [unicodedata.normalize("NFC", w)
            for w, t in tg.verse_tokens(verses[v - 1]) if t]


def build_verses(meta, classes, held, max_verses, rng):
    """Reassemble held-out verses in reading order from the shuffled corpus."""
    rows = defaultdict(list)
    for i, m in enumerate(meta):
        if str(m.get("parasha")) in held and m["taam"] in classes:
            rows[m["verse_ref"]].append(i)

    out = []
    refs = sorted(rows)
    rng.shuffle(refs)
    for ref in refs:
        words = verse_text(ref)
        if not words or not (4 <= len(words) <= 20):
            continue
        # one corpus row per text word, matched on the pointed form
        by_word = defaultdict(list)
        for i in rows[ref]:
            by_word[unicodedata.normalize("NFC", meta[i]["word_raw"])].append(i)
        if len(rows[ref]) != len(words):
            continue
        ordered = []
        for w in words:
            if not by_word.get(w):
                ordered = None
                break
            ordered.append(by_word[w].pop(0))
        if ordered:
            out.append((ref, ordered))
        if len(out) >= max_verses:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", type=int, default=250)
    args = ap.parse_args()

    srv = load_server()
    X = np.load(P / "features_v3_word" / "BEN13R_X.npy", mmap_mode="r")
    meta = json.loads((P / "features_v3_word" / "BEN13R_meta.json")
                      .read_text(encoding="utf-8"))
    classes = set(srv.get_model()["class_names"])

    rng = np.random.default_rng(42)
    par = np.unique([str(m.get("parasha", "?")) for m in meta])
    held = set(rng.choice(par, size=8, replace=False).tolist())

    verses = build_verses(meta, classes, held, args.verses, rng)
    print(f"  {len(verses)} פסוקים שלמים שוחזרו מ-{len(held)} פרשות מוחזקות\n")
    if not verses:
        print("  לא שוחזר אף פסוק — אין מה למדוד.")
        return

    # one pass: for every word, the isolated score and the in-phrase
    # probability, for the true ta'am and for a plausible wrong one
    rows = []
    for ref, idxs in verses:
        specs = [np.array(X[i]) for i in idxs]
        ctx = srv.context_probabilities(specs)
        if ctx is None:
            continue
        for k, i in enumerate(idxs):
            true_t = meta[i]["taam"]
            _, wp_true = srv.verify_match(specs[k], true_t)
            cp_true = ctx[k].get(true_t)
            # a realistic wrong answer: what the model itself most nearly
            # confused this word with, rather than a random ta'am
            alt = sorted((p, t) for t, p in ctx[k].items() if t != true_t)
            if not alt:
                continue
            wrong = alt[-1][1]
            _, wp_wrong = srv.verify_match(specs[k], wrong)
            cp_wrong = ctx[k].get(wrong)
            if None in (cp_true, cp_wrong):
                continue
            rows.append((wp_true, cp_true, wp_wrong, cp_wrong))

    if not rows:
        print("  מודל הרצף לא זמין — אין מה למדוד.")
        return
    print(f"  {len(rows):,} מילים\n")
    print(f"  {'משקל הקשר':<12}{'סותר נכון':<14}{'תופס שגוי':<14}הפרדה")

    best = None
    for W in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        def comb(w, c):
            return (max(w, 1e-6) ** (1 - W)) * (max(c, 1e-6) ** W)
        good = np.array([comb(r[0], r[1]) for r in rows])
        bad = np.array([comb(r[2], r[3]) for r in rows])
        thr = float(np.percentile(good, CONTRADICT_BUDGET * 100))
        catch = float((bad < thr).mean())
        contra = float((good < thr).mean())
        sep = float(good.mean() - bad.mean())
        mark = ""
        if best is None or catch > best[1]:
            best = (W, catch, thr)
            mark = "  ←"
        print(f"  {W:<12.2f}{contra*100:<14.1f}{catch*100:<14.1f}{sep:+.3f}{mark}")

    base = None
    for W in (0.0,):
        def comb(w, c):
            return w
        good = np.array([r[0] for r in rows])
        bad = np.array([r[2] for r in rows])
        thr = float(np.percentile(good, CONTRADICT_BUDGET * 100))
        base = float((bad < thr).mean())

    print(f"\n  מילה בודדת (משקל 0):  תופס {base*100:.1f}%")
    print(f"  הטוב ביותר (משקל {best[0]}): תופס {best[1]*100:.1f}%")
    delta = (best[1] - base) * 100
    print(f"  שינוי: {delta:+.1f} נקודות באותו תקציב "
          f"({CONTRADICT_BUDGET*100:.0f}% סותר נכון)")
    if delta < 2.0:
        print("\n  ⚠ פחות משתי נקודות — בתחום הרעש. ההקשר אינו מצדיק את "
              "עצמו כאן, וכדאי להשאיר את המשקל על 0.")

    out = P / "results" / "calibration" / "context_weight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "best_weight": best[0], "catch_at_budget": best[1],
        "baseline_word_only": base, "delta_points": round(delta, 2),
        "contradict_budget": CONTRADICT_BUDGET,
        "n_words": len(rows), "n_verses": len(verses),
        "held_out_parashot": sorted(held),
    }, indent=2), encoding="utf-8")
    print(f"\n  נשמר: {out}")


if __name__ == "__main__":
    main()
