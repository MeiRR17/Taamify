"""Bind the existing keep-ok calibration to the current local checkpoints.

This is a present-day snapshot, not proof of historical model provenance.
Paired API replay is still required before any release. No serving changes.
"""
import argparse
import json
from pathlib import Path

from webapp import server
from webapp.channel_calibration import policy_digest, sha256_file, validate_policy


def prepare(source, rows):
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if sha256_file(rows) != evidence["rows_sha256"]:
        raise ValueError("source calibration rows digest mismatch")
    policy = {
        "schema_version": 1, "status": "candidate",
        "source_artifact_sha256": sha256_file(source),
        "source_rows_sha256": evidence["rows_sha256"],
        "binding_note": "current checkpoint snapshot; historical rows were not model-hashed",
        "wrong_score": "rounded_api_confidence_3dp",
        "minimum_confidence": server.get_severity().CHANNEL_MIN_CONFIDENCE,
        "frontend": server.channel_frontend(),
        "bindings": server.channel_bindings(),
        "thresholds": evidence["thresholds"],
    }
    policy["policy_sha256"] = policy_digest(policy)
    return validate_policy(policy, policy["bindings"], policy["frontend"],
                           policy["minimum_confidence"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=server.PROJECT_DIR /
                    "results/calibration/channel_calibration_candidate_keepok.json")
    ap.add_argument("--rows", type=Path, default=server.PROJECT_DIR /
                    "results/research_os/channel_cal/rows.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    policy = prepare(args.source, args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing reviewed candidate.
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(policy, indent=2, ensure_ascii=False) + "\n")
    print(f"candidate sha256: {policy['policy_sha256']}")


if __name__ == "__main__":
    main()
