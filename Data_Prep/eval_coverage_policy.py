"""
Does withholding a verdict on unsupported words fix the false accusations?
==========================================================================
FINDINGS.md 12 established that the live path contradicts ~14% of correctly
read words while calibration promises 1.6%, and that the gap tracks whether
the word had training support at all: ~6% for words in the corpus, ~20% for
words absent from it.

That first pass matched words by exact raw text, which conflates a genuine
build-time drop with a maqaf or normalisation difference. This redoes it with
the corpus builder's own `normalize_word`, so "absent" means absent, and then
asks the question the app actually needs answered:

  if the app stayed silent on words it has no support for, what would the
  false-accusation rate become, and how much of the reading would go
  unjudged?

A verdict withheld is not free -- a learner practising a phrase where a third
of the words come back grey learns less. So both numbers are reported, and
the tradeoff is left visible rather than resolved here.

Usage:
  python Data_Prep/eval_coverage_policy.py --phrases 120
"""
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEAT = PROJECT_DIR / "features_v3_word"
READING = PROJECT_DIR / "data" / "ben13_reading"
OUT = PROJECT_DIR / "results" / "coverage"
API = "http://127.0.0.1:8000"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", type=int, default=120)
    args = ap.parse_args()

    tg = load("tg", "Data_Prep/taam_grammar.py")
    cv = load("cv", "Data_Prep/clean_align_v2.py")

    meta = json.loads((FEAT / "BEN13R_meta.json").read_text(encoding="utf-8"))
    # Support is keyed the way the corpus itself keys a word: verse plus the
    # normalised spelling. Raw text differs by maqaf and pointing details that
    # have nothing to do with whether the audio was learned.
    supported = {(m["parasha"], m["verse_ref"], cv.normalize_word(m["word_raw"]))
                 for m in meta}
    print(f"corpus support: {len(supported):,} distinct (verse, word) keys")

    rows = json.loads((READING / "maintext.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    rng.shuffle(rows)

    buckets = {"supported": [0, 0], "unsupported": [0, 0]}
    n = 0
    for r in rows:
        if str(r.get("block")) in ("-1", "0") or not r.get("url"):
            continue
        path = READING / "yerushalmi" / r["url"]
        if not path.exists():
            continue
        toks = [(w, t) for w, t in tg.verse_tokens(r["text"]) if t]
        if len(toks) < 2:
            continue
        words = [w for w, _ in toks]
        taamim = [t for _, t in toks]
        out = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{API}/api/check_phrase",
             "-F", f"audio=@{path}",
             "-F", f"words={json.dumps(words, ensure_ascii=False)}",
             "-F", f"taamim={json.dumps(taamim)}"],
            capture_output=True, text=True).stdout
        try:
            d = json.loads(out)
        except ValueError:
            continue
        if not d.get("aligned"):
            continue
        vref = f"{r['book']}.{r['perek']}.{r['pasuk']}"
        for w in d.get("words", []):
            if w["status"] not in ("ok", "wrong"):
                continue
            key = (str(r["parasha"]), vref, cv.normalize_word(w["word"]))
            b = "supported" if key in supported else "unsupported"
            buckets[b][0] += 1
            buckets[b][1] += 1 if w["status"] == "wrong" else 0
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{args.phrases}", flush=True)
        if n >= args.phrases:
            break

    sup_n, sup_bad = buckets["supported"]
    uns_n, uns_bad = buckets["unsupported"]
    total, total_bad = sup_n + uns_n, sup_bad + uns_bad

    L = []
    def say(s=""):
        L.append(s); print(s)

    say("=" * 70)
    say("  Coverage and the withhold-a-verdict policy")
    say("=" * 70)
    say(f"\n  {n} phrases, {total} words judged. Every one read correctly by")
    say("  Ben13, so every 'wrong' is a false accusation.\n")
    say(f"  {'bucket':>14}{'words':>9}{'share':>9}{'FALSELY ACCUSED':>18}")
    say("  " + "-" * 50)
    for label, (bn, bbad) in (("supported", buckets["supported"]),
                              ("unsupported", buckets["unsupported"])):
        if bn:
            say(f"  {label:>14}{bn:>9}{bn/total*100:>8.1f}%{bbad/bn*100:>17.1f}%")
    say(f"  {'ALL':>14}{total:>9}{100.0:>8.1f}%{total_bad/total*100:>17.1f}%")

    say("\n  policy: stay silent on unsupported words")
    say(f"    false accusations: {total_bad/total*100:.1f}%  ->  "
        f"{sup_bad/sup_n*100:.1f}%" if sup_n else "")
    say(f"    cost: {uns_n/total*100:.1f}% of the reading comes back unjudged")
    say("\n" + "=" * 70)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "coverage.json").write_text(json.dumps({
        "phrases": n, "supported": buckets["supported"],
        "unsupported": buckets["unsupported"],
    }, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT}/coverage.json")


if __name__ == "__main__":
    main()
