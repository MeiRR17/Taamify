"""
Task 1 -- a calibration CANDIDATE for the four grammar channels.
================================================================

The ta'am channel decides from percentile-derived thresholds. The grammar
channels (stress / dagesh / guttural / sheva) still decide from one flat
rule in run_channels(): confidence < CHANNEL_MIN_CONFIDENCE (0.9) is
"unsure", otherwise argmax right/wrong. Measured on parasha 5 only (the old
first-300-blocks sample), that rule says "wrong" to 5.6-8.1% of correct
words, against ta'am's 2.0%.

This derives per-channel thresholds the way ta'am's were derived, and fixes
the two problems the S0 audit found with how that was done:

  * Calibration and evaluation are DISJOINT. Inside every one of the eight
    held-out parashot, verses are split in two by a recorded seed; a verse
    (and every block in it) lands on one side only, so neighbouring blocks
    sharing a breath cannot straddle the split.
  * Both sides are drawn from ALL eight parashot, equally.

Everything goes through the live API (/api/check_phrase), as the product
runs it. All four channels are binary, so the probability of the EXPECTED
label is recoverable exactly from what the API returns:
p_exp = confidence if heard == expected else 1 - confidence.

Decision rule per channel (same three states as ta'am):
    p_exp <  say_wrong_below      -> "wrong"
    p_exp >= say_correct_above    -> "ok"
    otherwise                     -> "unsure"
The current rule is the special case (0.1, 0.9).

say_wrong_below is the contradict-budget percentile of p_exp on correct
audio (calibration half). say_correct_above is set from the proxy error:
the same audio with the expected label flipped, whose p_exp is 1 - p_true.
For a binary channel that proxy is fully determined by the correct-audio
distribution -- it measures "the reader produced the other class", not an
authentic learner error, and is reported as a proxy only.

Nothing here touches the served thresholds. Output:
  results/research_os/channel_cal/rows.jsonl          raw per-word rows (local)
  results/calibration/channel_calibration_candidate.json   candidate + eval

Usage:
  venv/bin/python Data_Prep/calibrate_channels.py collect --per-parasha 50
  venv/bin/python Data_Prep/calibrate_channels.py analyze
"""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_end_to_end as C   # noqa: E402

P = Path(__file__).resolve().parent.parent
ROWS = P / "results" / "research_os" / "channel_cal" / "rows.jsonl"
CANDIDATE = P / "results" / "calibration" / "channel_calibration_candidate.json"
CHANNELS = ("stress", "dagesh", "guttural", "sheva")
CURRENT = {"say_wrong_below": 0.1, "say_correct_above": 0.9}   # CHANNEL_MIN_CONFIDENCE=0.9


def split_blocks(per_parasha, seed, min_words=2):
    """{parasha: {"cal": [...], "eval": [...]}}, split by verse, then sampled."""
    meta = json.loads((P / "features_v3_word" / "BEN13R_meta.json")
                      .read_text(encoding="utf-8"))
    man = json.loads((P / "data" / "ben13_reading" /
                      "manifest_yerushalmi.json").read_text(encoding="utf-8"))
    by_block = {(r["book"], r["perek"], r["pasuk"], str(r["block"])): r["local_path"]
                for r in man}
    blocks = defaultdict(list)
    for i, m in enumerate(meta):
        blocks[(m["verse_ref"], m["segment_block"])].append(i)
    held = set(C.HELD_OUT)
    by_verse = defaultdict(lambda: defaultdict(list))
    for (ref, blk), idxs in blocks.items():
        par = str(meta[idxs[0]].get("parasha"))
        if len(idxs) < min_words or par not in held:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["start"])
        b, c, v = ref.split(".")
        path = by_block.get((b, c, v, str(blk)))
        if not path or not (P / path).exists():
            continue
        by_verse[par][ref].append({
            "parasha": par, "verse": ref, "block": str(blk), "path": str(P / path),
            "words": [meta[i]["word_raw"] for i in idxs],
            "taamim": [meta[i]["taam"] for i in idxs]})
    rng = random.Random(seed)
    out = {}
    for par in sorted(by_verse, key=int):
        verses = sorted(by_verse[par])
        rng.shuffle(verses)
        half = len(verses) // 2
        sides = {"cal": verses[:half], "eval": verses[half:]}
        out[par] = {}
        for side, vs in sides.items():
            pool = [blk for v in vs for blk in by_verse[par][v]]
            out[par][side] = rng.sample(pool, min(per_parasha, len(pool)))
    return out


def collect(args):
    plan = split_blocks(args.per_parasha, args.seed)
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    n_blocks = n_gated = 0
    with ROWS.open("w", encoding="utf-8") as fh:
        for par, sides in plan.items():
            for side, blks in sides.items():
                for blk in blks:
                    n_blocks += 1
                    d = C.post(blk["path"], blk["words"], blk["taamim"])
                    base = {"split": side, "parasha": par, "verse": blk["verse"],
                            "block": blk["block"]}
                    if not d or not d.get("aligned"):
                        n_gated += 1
                        fh.write(json.dumps({**base, "gated": True}) + "\n")
                        continue
                    for wi, w in enumerate(d.get("words", [])):
                        for name, ch in (w.get("channels") or {}).items():
                            conf = ch.get("confidence")
                            if conf is None:
                                continue
                            p_exp = conf if ch["heard"] == ch["expected"] else 1 - conf
                            fh.write(json.dumps({
                                **base, "word_index": wi, "channel": name,
                                "expected": ch["expected"], "heard": ch["heard"],
                                "confidence": conf, "p_exp": round(p_exp, 4),
                                "served_status": ch.get("status"),
                                "taam_status": w.get("status")},
                                ensure_ascii=False) + "\n")
                    print(f"  {n_blocks} blocks, {n_gated} gated", end="\r", flush=True)
    print(f"\n  wrote {ROWS} -- {n_blocks} blocks, {n_gated} gated by alignment")


def decide(p, t):
    return np.where(p < t["say_wrong_below"], "wrong",
                    np.where(p >= t["say_correct_above"], "ok", "unsure"))


def rates(p, t):
    """On correct audio (p = p_exp) and on the flipped-label proxy (1 - p)."""
    on_correct, on_proxy = decide(p, t), decide(1 - p, t)
    return {"false_accusation": float((on_correct == "wrong").mean()),
            "coverage_correct": float((on_correct != "unsure").mean()),
            "proxy_catch": float((on_proxy == "wrong").mean()),
            "proxy_blessed": float((on_proxy == "ok").mean())}


def block_bootstrap_fa(rows, t, n_boot=2000, seed=0):
    """95% CI of false accusation, resampling whole blocks (words in a block
    share a recording, so they are not independent)."""
    by_block = defaultdict(list)
    for r in rows:
        by_block[(r["verse"], r["block"])].append(r["p_exp"])
    keys = list(by_block)
    wrong = np.array([(np.array(by_block[k]) < t["say_wrong_below"]).sum() for k in keys])
    total = np.array([len(by_block[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    est = wrong[idx].sum(1) / np.maximum(total[idx].sum(1), 1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def any_wrong_rate(rows, thresholds):
    """Product-level: share of correctly-read words where ANY grammar channel
    says 'wrong' (thresholds: {channel: t})."""
    words = defaultdict(bool)
    for r in rows:
        key = (r["verse"], r["block"], r["word_index"])
        words[key] |= bool(r["p_exp"] < thresholds[r["channel"]]["say_wrong_below"])
    return float(np.mean(list(words.values()))) if words else None


def analyze(args):
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines()]
    gated = defaultdict(lambda: [0, 0])
    for r in rows:
        if "channel" not in r:
            gated[r["split"]][0] += 1
    words = [r for r in rows if "channel" in r]
    cal = [r for r in words if r["split"] == "cal"]
    ev = [r for r in words if r["split"] == "eval"]
    overlap = {r["verse"] for r in cal} & {r["verse"] for r in ev}
    assert not overlap, f"calibration and evaluation share verses: {sorted(overlap)[:5]}"

    candidate, report = {}, {}
    for name in CHANNELS:
        pc = np.array([r["p_exp"] for r in cal if r["channel"] == name])
        pe = np.array([r["p_exp"] for r in ev if r["channel"] == name])
        if not len(pc) or not len(pe):
            continue
        # Strict "<" in decide(): taking the budget percentile itself as the
        # cut keeps the calibration-half rate at or under budget.
        wrong_below = float(np.percentile(pc, args.contradict_budget * 100))
        # "ok" must not be said to more than bless_budget of proxy errors,
        # whose score is 1 - p_true: the (1 - bless) percentile of 1 - pc.
        correct_above = float(np.percentile(1 - pc, (1 - args.bless_budget) * 100))
        if args.fixed_correct_above is not None:
            correct_above = args.fixed_correct_above
        correct_above = max(correct_above, wrong_below)
        t = {"say_wrong_below": round(wrong_below, 4),
             "say_correct_above": round(correct_above, 4)}
        candidate[name] = t
        ev_rows = [r for r in ev if r["channel"] == name]
        report[name] = {
            "n_cal": int(len(pc)), "n_eval": int(len(pe)),
            "eval_parashot": sorted({r["parasha"] for r in ev_rows}, key=int),
            "candidate": t,
            "before_eval": {**rates(pe, CURRENT),
                            "fa_ci95_blocks": block_bootstrap_fa(ev_rows, CURRENT)},
            "after_eval": {**rates(pe, t),
                           "fa_ci95_blocks": block_bootstrap_fa(ev_rows, t)},
            "after_cal": rates(pc, t),
            "after_eval_by_parasha": {
                par: rates(np.array([r["p_exp"] for r in ev_rows if r["parasha"] == par]), t)
                for par in sorted({r["parasha"] for r in ev_rows}, key=int)},
        }
    combined = {
        "before": any_wrong_rate(ev, {c: CURRENT for c in CHANNELS}),
        "after": any_wrong_rate(ev, {c: candidate.get(c, CURRENT) for c in CHANNELS}),
    }
    body = {"status": "candidate -- NOT served; webapp/server.py still uses "
                      "CHANNEL_MIN_CONFIDENCE",
            "rule": "p_exp < say_wrong_below -> wrong; >= say_correct_above -> ok; else unsure",
            "contradict_budget": args.contradict_budget,
            "bless_budget": args.bless_budget,
            "fixed_correct_above": args.fixed_correct_above,
            "split": "per parasha, verses split 50/50 by seed; cal and eval verse-disjoint",
            "held_out_parashot": sorted(C.HELD_OUT, key=int),
            "proxy_note": "proxy_catch/proxy_blessed flip the expected label on correct "
                          "audio; not authentic learner errors",
            "gated_blocks": {k: v[0] for k, v in gated.items()},
            "rows_sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest(),
            "thresholds": candidate,
            "combined_any_channel_wrong_on_correct_words_eval": combined,
            "channels": report}
    body["candidate_sha256"] = hashlib.sha256(
        json.dumps(candidate, sort_keys=True).encode()).hexdigest()
    args.out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    pct = lambda v: f"{v*100:5.1f}%"
    print(f"  {'channel':<9}{'n_eval':>7}   {'FA before':>10} {'FA after':>9}  "
          f"{'cov before':>10} {'cov after':>9}  {'catch after':>11}  thresholds")
    for name, r in report.items():
        b, a = r["before_eval"], r["after_eval"]
        print(f"  {name:<9}{r['n_eval']:>7}   {pct(b['false_accusation']):>10} "
              f"{pct(a['false_accusation']):>9}  {pct(b['coverage_correct']):>10} "
              f"{pct(a['coverage_correct']):>9}  {pct(a['proxy_catch']):>11}  "
              f"{r['candidate']['say_wrong_below']}/{r['candidate']['say_correct_above']}"
              f"   CI {pct(a['fa_ci95_blocks'][0])}-{pct(a['fa_ci95_blocks'][1])}")
    print(f"  any-channel 'wrong' on a correct word: {pct(combined['before'])} -> "
          f"{pct(combined['after'])}")
    print(f"  wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--per-parasha", type=int, default=50, help="blocks per parasha per half")
    c.add_argument("--seed", type=int, default=20261022)
    a = sub.add_parser("analyze")
    a.add_argument("--contradict-budget", type=float, default=0.02)
    a.add_argument("--bless-budget", type=float, default=0.10)
    a.add_argument("--fixed-correct-above", type=float, default=None,
                   help="keep say_correct_above here (0.9 = served) and move only "
                        "say_wrong_below, so nothing shifts toward 'ok'")
    a.add_argument("--out", type=Path, default=CANDIDATE)
    args = ap.parse_args()
    {"collect": collect, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
