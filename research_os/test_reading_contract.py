import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from webapp import server
from webapp.reading_contract import (CONSENT_SCOPES, CONSENT_VERSION, TRADITION,
                                    capture_metadata, phrase_inputs)


class ReadingContractTests(unittest.TestCase):
    def test_explicit_missing_runtime_bundle_rejects_startup(self):
        import os
        with patch.dict(os.environ, {'TAAMIFY_RUNTIME_BUNDLE': 'results/missing-test-only-bundle.json'}):
            with self.assertRaises(server.runtime_binding.RuntimeBundleError):
                with TestClient(server.app):
                    self.fail('missing bundle accepted traffic')
        server.get_runtime_bundle.cache_clear()

    def test_dagesh_stays_attached_under_mark_permutations(self):
        for word in ['\u05d1\u05b8\u05bc\u0591\u05e8', '\u05d1\u0591\u05b8\u05bc\u05e8',
                     '\u05d1\u05b8\u0591\u05bc\u05e8']:
            self.assertEqual(server.expected_channel_label('dagesh', word), 'dagesh')
        self.assertEqual(server.expected_channel_label('dagesh', 'בָר'), 'no_dagesh')
        self.assertIsNone(server.expected_channel_label('dagesh', 'בָּכָּה'))

    def test_phrase_validation_preserves_position(self):
        self.assertEqual(phrase_inputs('["א", "ב"]', '[null, "Mercha"]'),
                         (['א', 'ב'], [None, 'Mercha']))
        for w, t in [('[]', '[]'), ('["א", ""]','["Mercha","Mercha"]'),
                     ('[1]', '[null]'), ('["א"]','[{}]'), ('["א"]','[]')]:
            with self.assertRaises(ValueError):
                phrase_inputs(w, t)

    def test_capture_needs_explicit_complete_consent(self):
        fields = ['true', 'voice_12345', 'session_12345', CONSENT_VERSION,
                  json.dumps(CONSENT_SCOPES), TRADITION]
        self.assertTrue(capture_metadata(*fields)['consent'])
        for i, invalid in [(0,'maybe'),(1,'anon'),(2,''),(3,'old'),(4,'[]'),(5,'ashkenazi')]:
            changed = list(fields); changed[i] = invalid
            with self.assertRaises(ValueError):
                capture_metadata(*changed)
        fields[0] = 'false'
        self.assertIsNone(capture_metadata(*fields))

    def test_bad_requests_do_not_reach_models_or_write_recordings(self):
        with patch.object(server, 'get_model', side_effect=AssertionError('model called')):
            with TestClient(server.app) as client:
                for words in ['[null]', '[1]', '[""]', '[]']:
                    response = client.post('/api/check_phrase', data={'words': words, 'taamim':'[null]'},
                        files={'audio':('input.wav', b'not-wav', 'audio/wav')})
                    self.assertEqual(response.status_code, 400)
                response = client.post('/api/check_phrase', data={'words':'["א"]','taamim':'[null]'},
                    files={'audio':('input.wav', b'not-wav', 'audio/wav')})
                self.assertEqual(response.status_code, 400)

    def test_capture_records_rejected_clear_silence_without_gold_labels(self):
        data = io.BytesIO(); sf.write(data, np.zeros(16000), 16000, format='WAV')
        with tempfile.TemporaryDirectory() as root, \
             patch.object(server,'PHRASES_DIR',Path(root)), \
             patch.object(server,'PHRASES_LOG',Path(root)/'phrases_log.jsonl'):
            with TestClient(server.app) as client:
                fields={'words':'["א"]','taamim':'["Mercha"]','consent':'true',
                        'voice_id':'voice_12345','session_id':'session_12345',
                        'consent_version':CONSENT_VERSION, 'consent_scope':json.dumps(CONSENT_SCOPES),
                        'tradition':TRADITION}
                response=client.post('/api/check_phrase', data=fields,
                    files={'audio':('silence.wav',data.getvalue(),'audio/wav')})
                self.assertEqual(response.status_code,200)
                result=response.json()
                self.assertFalse(result['aligned'])
                self.assertEqual(result['input_quality']['reason'],'silence')
                self.assertTrue(result['saved_clip'])
                rows=[json.loads(s) for s in (Path(root)/'phrases_log.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['schema_version'],2)
                self.assertNotIn('performed_taamim',rows[0])
                self.assertEqual(rows[0]['taamim'],['Mercha'])
                self.assertTrue(rows[0]['audio_sha256'])
                fields['consent']='false'
                client.post('/api/check_phrase',data=fields,
                    files={'audio':('silence.wav',data.getvalue(),'audio/wav')})
                self.assertEqual(len(list(Path(root).glob('*.wav'))),1)


if __name__ == '__main__':
    unittest.main()
