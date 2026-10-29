"""Paired local HTTP regression on the original, verse-disjoint eval side.

Two already-started local servers are required. Audio and detailed results
stay local. This is an integration check, not an independent speaker test.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import requests

from Data_Prep.calibrate_channels import split_blocks


def local_url(value):
    parsed = urlparse(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("only local HTTP servers are allowed")
    return value.rstrip("/")


def post(session, base, block):
    with Path(block["path"]).open("rb") as stream:
        response = session.post(base + "/api/check_phrase",
                                files={"audio": ("clip.mp3", stream, "audio/mpeg")},
                                data={"words": json.dumps(block["words"], ensure_ascii=False),
                                      "taamim": json.dumps(block["taamim"]), "consent": "false"},
                                timeout=120)
    response.raise_for_status()
    result = response.json()
    if "error" in result or "aligned" not in result:
        raise RuntimeError(f"invalid phrase response: {result}")
    return result


def compare(before, after, counts):
    assert before["aligned"] == after["aligned"], "alignment gate changed"
    if not before["aligned"]:
        counts["gated"] += 1
        return
    assert "channel_calibration" not in before, "baseline used a candidate"
    assert after.get("channel_calibration", {}).get("mode") == "candidate", \
        "candidate provenance missing from phrase response"
    assert before["align_score"] == after["align_score"], "alignment score changed"
    assert len(before["words"]) == len(after["words"]), "word count changed"
    assert before["severity"]["should_repeat"] == after["severity"]["should_repeat"], \
        "repeat guard changed"
    counts["aligned"] += 1
    for old, new in zip(before["words"], after["words"]):
        for key in ("word", "expected", "status", "confidence", "heard", "start", "end"):
            assert old.get(key) == new.get(key), f"taam/crop changed: {key}"
        old_ch, new_ch = old.get("channels", {}), new.get("channels", {})
        assert old_ch.keys() == new_ch.keys(), "channel coverage changed"
        counts["words"] += 1
        if old_ch:
            counts["applicable_words"] += 1
            counts["any_wrong_before"] += any(c["status"] == "wrong" for c in old_ch.values())
            counts["any_wrong_after"] += any(c["status"] == "wrong" for c in new_ch.values())
        for name, a in old_ch.items():
            b = new_ch[name]
            for key in ("expected", "heard", "confidence"):
                assert a[key] == b[key], f"channel inference changed: {name}/{key}"
            transition = (a["status"], b["status"])
            assert transition in {("ok", "ok"), ("wrong", "wrong"),
                                   ("wrong", "unsure"), ("unsure", "unsure")}, transition
            counts[f"{name}:{a['status']}->{b['status']}"] += 1
            if b["status"] != "wrong":
                assert "message" not in b and "severity" not in b, "stale warning"
        if (not new["severities"] and after["report"]["headline"] == "הקטע נקרא נכון"
                and any(c["status"] == "unsure" for c in new_ch.values())):
            raise AssertionError("uncertain grammar presented as fully correct")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=local_url, required=True)
    ap.add_argument("--candidate", type=local_url, required=True)
    ap.add_argument("--per-parasha", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.per_parasha <= 50 or args.baseline == args.candidate:
        ap.error("use distinct servers and 1..50 blocks per parasha")
    session = requests.Session()
    session.trust_env = False  # local audio must never go through an HTTP proxy
    configs = {}
    for name, url in [("baseline", args.baseline), ("candidate", args.candidate)]:
        response = session.get(url + "/api/channel_config", timeout=120)
        response.raise_for_status()
        configs[name] = response.json()
    assert configs["baseline"]["mode"] == "legacy"
    assert configs["candidate"]["mode"] == "candidate"
    plan = split_blocks(50, 20261022)
    per_parasha, total = defaultdict(Counter), Counter()
    start = time.monotonic()
    for par, halves in plan.items():
        for block in halves["eval"][:args.per_parasha]:
            before = post(session, args.baseline, block)
            after = post(session, args.candidate, block)
            compare(before, after, per_parasha[par])
            per_parasha[par]["requests"] += 2
        total.update(per_parasha[par])
        print(f"parasha {par}: {dict(per_parasha[par])}", flush=True)
    report = {"status": "passed_integration_only", "seed": 20261022,
              "selection": "first N sampled eval blocks from original 50/50 verse split",
              "per_parasha": args.per_parasha, "configs": configs,
              "counts": dict(total), "by_parasha": dict(per_parasha),
              "elapsed_seconds": round(time.monotonic() - start, 2),
              "limitations": ["single Ben13 voice; no authentic learner errors",
                              "integration regression, not an independent accuracy estimate"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
