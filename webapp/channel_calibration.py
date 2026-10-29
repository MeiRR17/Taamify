"""Explicit, checkpoint-bound grammar calibration candidates. No model I/O."""
import hashlib
import json
import math
from pathlib import Path


class CalibrationError(ValueError):
    """A selected candidate cannot describe this inference pipeline."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_digest(policy):
    body = {k: v for k, v in policy.items() if k != "policy_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False)
                          .encode("utf-8")).hexdigest()


def validate_policy(policy, bindings, frontend, minimum_confidence):
    """Fail closed on changed models, label order, crops or decision policy.

    Only the keep-ok candidate is supported: an existing 'wrong' may become
    'unsure'; no prediction may become a new 'ok' or a new 'wrong'.
    """
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise CalibrationError("unsupported channel calibration schema")
    if policy.get("status") != "candidate":
        raise CalibrationError("channel calibration must be an explicit candidate")
    try:
        digest = policy_digest(policy)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("invalid channel calibration payload") from exc
    if policy.get("policy_sha256") != digest:
        raise CalibrationError("channel calibration digest mismatch")
    if policy.get("frontend") != frontend:
        raise CalibrationError("channel calibration frontend/crop mismatch")
    if policy.get("bindings") != bindings:
        raise CalibrationError("channel calibration checkpoint/classes mismatch")
    if policy.get("minimum_confidence") != minimum_confidence:
        raise CalibrationError("channel calibration confidence policy mismatch")
    if policy.get("wrong_score") != "rounded_api_confidence_3dp":
        raise CalibrationError("unsupported channel calibration score")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(bindings):
        raise CalibrationError("channel calibration must cover every channel")
    for name, values in thresholds.items():
        if not isinstance(values, dict):
            raise CalibrationError(f"invalid thresholds for {name}")
        lo, hi = values.get("say_wrong_below"), values.get("say_correct_above")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (lo, hi)):
            raise CalibrationError(f"non-finite thresholds for {name}")
        if not 0 <= lo <= round(1 - minimum_confidence, 10) or hi != minimum_confidence:
            raise CalibrationError(f"thresholds must preserve existing ok decisions: {name}")
    return policy


def load_policy(path, bindings, frontend, minimum_confidence):
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError("cannot read selected channel calibration") from exc
    return validate_policy(policy, bindings, frontend, minimum_confidence)


def channel_status(heard, expected, confidence, minimum_confidence, thresholds=None):
    """Preserve legacy decisions unless a candidate abstains from 'wrong'.

    Calibration rows were collected from API confidence rounded to 3 decimal
    places. Use that same score for the new wrong cut, while preserving the
    legacy unrounded confidence check (including its inclusive boundary).
    """
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return "unsure"
    if confidence < minimum_confidence:
        return "unsure"
    if heard == expected:
        return "ok"
    if thresholds is not None:
        p_expected = round(1 - round(confidence, 3), 4)
        if p_expected >= thresholds["say_wrong_below"]:
            return "unsure"
    return "wrong"
