"""API startup and feedback contracts; model loading is deliberately mocked."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from webapp import server
from webapp.channel_calibration import CalibrationError, policy_digest


class ChannelServerTests(unittest.TestCase):
    def tearDown(self):
        server.get_channel_calibration.cache_clear()

    def test_default_startup_does_not_load_candidate_models(self):
        with patch.dict(os.environ):
            os.environ.pop("TAAMIFY_CHANNEL_CALIBRATION", None)
            server.get_channel_calibration.cache_clear()
            with patch.object(server, "channel_bindings") as binding:
                with TestClient(server.app) as client:
                    response = client.get("/api/channel_config")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["mode"], "legacy")
                binding.assert_not_called()

    def test_invalid_explicit_candidate_fails_startup(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.json"
            path.write_text("{}")
            with patch.dict(os.environ, {"TAAMIFY_CHANNEL_CALIBRATION": str(path)}), \
                    patch.object(server, "channel_bindings", return_value={}):
                server.get_channel_calibration.cache_clear()
                with self.assertRaises(CalibrationError):
                    with TestClient(server.app):
                        self.fail("an invalid candidate accepted traffic")

    def test_candidate_digest_is_visible_without_local_paths(self):
        policy = {"schema_version": 1, "status": "candidate", "bindings": {},
                  "frontend": server.channel_frontend(), "minimum_confidence": 0.9,
                  "wrong_score": "rounded_api_confidence_3dp", "thresholds": {}}
        policy["policy_sha256"] = policy_digest(policy)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "candidate.json"
            path.write_text(json.dumps(policy))
            with patch.dict(os.environ, {"TAAMIFY_CHANNEL_CALIBRATION": str(path)}), \
                    patch.object(server, "channel_bindings", return_value={}):
                server.get_channel_calibration.cache_clear()
                with TestClient(server.app) as client:
                    result = client.get("/api/channel_config").json()
                    self.assertEqual(result["mode"], "candidate")
                    self.assertEqual(result["policy_sha256"], policy["policy_sha256"])
                    self.assertNotIn(root, json.dumps(result))

    def test_uncertain_candidate_channels_do_not_confirm_whole_phrase(self):
        words = [{"status": "ok", "severities": [],
                  "channels": {"stress": {"status": "unsure"}}}]
        result = server.phrase_report(words, {"should_repeat": False}, calibrated_channels=True)
        self.assertNotEqual(result["headline"], "הקטע נקרא נכון")
        self.assertFalse(result["should_repeat"])
        legacy = server.phrase_report(words, {"should_repeat": False})
        self.assertNotEqual(legacy["headline"], "הקטע נקרא נכון")

    def test_partial_coverage_does_not_confirm_whole_phrase(self):
        words = [{"status": "ok", "severities": []}, {"status": "unsupported"}]
        result = server.phrase_report(words, {"should_repeat": False})
        self.assertNotEqual(result["headline"], "הקטע נקרא נכון")
        self.assertFalse(result["should_repeat"])


if __name__ == "__main__":
    unittest.main()
