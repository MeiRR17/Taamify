"""Release-critical calibration guards, without loading a model or audio."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from webapp.channel_calibration import (
    CalibrationError, channel_status, load_policy, policy_digest, validate_policy,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.bindings = {"stress": {"checkpoint_sha256": "a" * 64,
                                    "class_names": ["milra", "milel"]}}
        self.frontend = {"sample_rate": 16000, "windows": {"stress": [2.0, 0.4]}}
        self.policy = {
            "schema_version": 1, "status": "candidate",
            "wrong_score": "rounded_api_confidence_3dp", "minimum_confidence": 0.9,
            "bindings": copy.deepcopy(self.bindings), "frontend": copy.deepcopy(self.frontend),
            "thresholds": {"stress": {"say_wrong_below": 0.014, "say_correct_above": 0.9}},
        }
        self.sign()

    def sign(self):
        self.policy["policy_sha256"] = policy_digest(self.policy)

    def validate(self):
        return validate_policy(self.policy, self.bindings, self.frontend, 0.9)

    def test_valid_policy_and_file_loading(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "candidate.json"
            path.write_text(json.dumps(self.policy))
            self.assertEqual(load_policy(path, self.bindings, self.frontend, 0.9), self.policy)

    def test_checkpoint_and_label_order_are_both_bound(self):
        for key, value in [("checkpoint_sha256", "b" * 64),
                           ("class_names", ["milel", "milra"])]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.policy)
                self.policy["bindings"]["stress"][key] = value
                self.sign()
                with self.assertRaisesRegex(CalibrationError, "checkpoint/classes"):
                    self.validate()
                self.policy = original

    def test_changed_frontend_or_missing_channel_is_rejected(self):
        self.policy["frontend"]["windows"]["stress"][0] = 1.0
        self.sign()
        with self.assertRaisesRegex(CalibrationError, "frontend/crop"):
            self.validate()
        self.policy["frontend"] = copy.deepcopy(self.frontend)
        self.policy["thresholds"] = {}
        self.sign()
        with self.assertRaisesRegex(CalibrationError, "every channel"):
            self.validate()

    def test_threshold_mutation_and_permissive_ok_are_rejected(self):
        self.policy["thresholds"]["stress"]["say_wrong_below"] = 0.02
        with self.assertRaisesRegex(CalibrationError, "digest"):
            self.validate()
        for key, value in [("say_wrong_below", 0.11), ("say_wrong_below", -0.01),
                           ("say_wrong_below", True), ("say_correct_above", 0.8)]:
            with self.subTest(key=key, value=value):
                self.policy["thresholds"]["stress"] = {
                    "say_wrong_below": 0.014, "say_correct_above": 0.9, key: value}
                self.sign()
                with self.assertRaises(CalibrationError):
                    self.validate()

    def test_non_finite_invalid_or_missing_payload_fails_closed(self):
        for value in (float("nan"), float("inf")):
            self.policy["thresholds"]["stress"]["say_wrong_below"] = value
            with self.assertRaises(CalibrationError):
                self.validate()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.json"
            for data in ("not JSON", "null", "[]"):
                path.write_text(data)
                with self.assertRaises(CalibrationError):
                    load_policy(path, self.bindings, self.frontend, 0.9)
            with self.assertRaises(CalibrationError):
                load_policy(path.with_name("missing.json"), self.bindings, self.frontend, 0.9)

    def test_legacy_inclusive_boundary_and_rounded_candidate_boundary(self):
        self.assertEqual(channel_status("a", "a", 0.8999, 0.9), "unsure")
        self.assertEqual(channel_status("a", "a", 0.9, 0.9), "ok")
        self.assertEqual(channel_status("a", "b", 0.9, 0.9), "wrong")
        thresholds = self.policy["thresholds"]["stress"]
        self.assertEqual(channel_status("a", "b", 0.986, 0.9, thresholds), "unsure")
        self.assertEqual(channel_status("a", "b", 0.987, 0.9, thresholds), "wrong")

    def test_candidate_never_promotes_to_ok_or_adds_wrong(self):
        thresholds = self.policy["thresholds"]["stress"]
        for i in range(10001):
            confidence = i / 10000
            for heard in ("expected", "other"):
                old = channel_status(heard, "expected", confidence, 0.9)
                new = channel_status(heard, "expected", confidence, 0.9, thresholds)
                self.assertIn((old, new), {("ok", "ok"), ("wrong", "wrong"),
                                          ("wrong", "unsure"), ("unsure", "unsure")})


if __name__ == "__main__":
    unittest.main()
