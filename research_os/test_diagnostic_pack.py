"""Synthetic fixture checks for local source resolution and review blinding."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from Data_Prep.holdout import HELD_OUT
from research_os.build_diagnostic_pack import build_pack
from research_os.source_audio import SourceConflict, SourceResolver


class DiagnosticPackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'features_v3_word').mkdir()
        (self.root / 'data/ben13_reading').mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def write_audio(self, name, frequency=210):
        path = self.root / 'data/ben13_reading' / name
        t = np.arange(48000) / 16000
        sf.write(path, .1 * np.sin(2 * np.pi * frequency * t), 16000, subtype='PCM_16')
        return str(path.relative_to(self.root))

    def source_row(self, path, par='5'):
        return {'book': '1', 'perek': par, 'pasuk': '1', 'block': '1',
                'parasha': par, 'aliya': '1', 'id': par, 'local_path': path,
                'text': 'וַיָּבֹ֤א הָאִישׁ֙ הַבַּ֔יְתָה'}

    def fixture(self):
        manifest, meta = [], []
        for index, par in enumerate(HELD_OUT):
            path = self.write_audio(par + '.wav', frequency=210 + 20 * index)
            source = self.source_row(path, par)
            manifest.append(source)
            for wi, (word, taam) in enumerate(zip(source['text'].split(),
                                                 ['Mahapakh', 'Pashta', 'Zaqef_Qatan'])):
                meta.append({'word_raw': word, 'taam': taam, 'parasha': par,
                             'verse_ref': f'1.{par}.1', 'segment_block': '1',
                             'start': wi * .9, 'end': (wi + 1) * .9,
                             'block_pos': wi, 'block_size': 3,
                             'read_in_sequence': True, 'low_confidence': False})
        (self.root / 'features_v3_word/BEN13R_meta.json').write_text(json.dumps(meta))
        (self.root / 'data/ben13_reading/manifest_yerushalmi.json').write_text(json.dumps(manifest))

    def test_equal_paths_are_not_required_but_equal_bytes_and_text_are(self):
        first = self.write_audio('a.wav')
        second = 'data/ben13_reading/b.wav'
        (self.root / second).write_bytes((self.root / first).read_bytes())
        resolver = SourceResolver(self.root, [self.source_row(first), self.source_row(second)])
        source = resolver.resolve(('1', '5', '1', '1'))
        self.assertEqual(len(source['relative_paths']), 2)
        self.assertEqual(resolver.report()['selected_key_checks']['byte_identical_path_collision_keys'], 1)

    def test_conflicting_audio_is_quarantined(self):
        first, second = self.write_audio('a.wav', 210), self.write_audio('b.wav', 330)
        resolver = SourceResolver(self.root, [self.source_row(first), self.source_row(second)])
        with self.assertRaisesRegex(SourceConflict, 'audio_content_conflict'):
            resolver.resolve(('1', '5', '1', '1'))

    def test_conflicting_text_is_quarantined_even_with_same_audio(self):
        path = self.write_audio('a.wav')
        a, b = self.source_row(path), self.source_row(path)
        b['text'] = 'הָאִישׁ֙'
        with self.assertRaisesRegex(SourceConflict, 'text_conflict'):
            SourceResolver(self.root, [a, b]).resolve(('1', '5', '1', '1'))

    def test_missing_source_is_not_replaced_by_duration_match(self):
        path = self.write_audio('unrelated.wav')
        resolver = SourceResolver(self.root, [self.source_row(path, '6')])
        with self.assertRaisesRegex(SourceConflict, 'source_key_missing'):
            resolver.resolve(('1', '5', '1', '1'))

    def test_real_packet_contract_has_empty_performed_labels_and_stratified_phrases(self):
        self.fixture()
        out = self.root / 'results/packet'
        report = build_pack(self.root, out, max_words=24, per_parasha=1)
        reviews = [json.loads(line) for line in (out / 'review_manifest.jsonl').read_text().splitlines()]
        benchmark = [json.loads(line) for line in (out / 'benchmark_manifest.jsonl').read_text().splitlines()]
        self.assertEqual(report['review_units'], 24)
        self.assertEqual(report['benchmark_phrases'], 8)
        self.assertEqual(set(report['benchmark_parashot']), set(HELD_OUT))
        self.assertTrue(all(row['performed_taam'] is None and row['human_correct'] is None for row in reviews))
        self.assertTrue(all(row['contiguous_original_phrase'] and not row['spliced'] for row in benchmark))
        self.assertTrue(all(Path(row['path']).is_file() for row in reviews + benchmark))
        page = (out / 'review.html').read_text()
        self.assertIn('id="expected" hidden', page)
        self.assertIn('blind_locked_at', page)
        self.assertNotIn('fetch(', page)
        self.assertNotIn('XMLHttpRequest', page)
        self.assertNotIn('model_predictions', page)
        with self.assertRaises(FileExistsError):
            build_pack(self.root, out)

    def test_selection_ids_and_accounting_are_deterministic(self):
        self.fixture()
        first, second = self.root / 'results/first', self.root / 'results/second'
        a = build_pack(self.root, first, seed=17, max_words=16, per_parasha=1)
        b = build_pack(self.root, second, seed=17, max_words=16, per_parasha=1)
        get_ids = lambda p: [json.loads(line)['sample_id'] for line in (p / 'review_manifest.jsonl').read_text().splitlines()]
        self.assertEqual(get_ids(first), get_ids(second))
        self.assertEqual(a['selection_accounting'], b['selection_accounting'])

    def test_benchmark_block_allowlist_limits_phrases_and_is_recorded(self):
        self.fixture()
        allowed = [f'1.{HELD_OUT[0]}.1:1', f'1.{HELD_OUT[1]}.1:1']
        report = build_pack(self.root, self.root / 'results/allow', max_words=1,
                            per_parasha=1, benchmark_blocks=allowed)
        benchmark = [json.loads(line) for line in
                     (self.root / 'results/allow/benchmark_manifest.jsonl').read_text().splitlines()]
        self.assertEqual(sorted(f"{row['verse']}:{row['block']}" for row in benchmark), sorted(allowed))
        self.assertEqual(report['benchmark_block_allowlist']['blocks'], 2)
        default = build_pack(self.root, self.root / 'results/default', max_words=1, per_parasha=1)
        self.assertIsNone(default['benchmark_block_allowlist'])
        self.assertEqual(default['benchmark_phrases'], len(HELD_OUT))

    def test_output_outside_ignored_results_is_rejected(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, 'ignored results'):
            build_pack(self.root, self.root / 'public')
        self.assertFalse((self.root / 'public').exists())

    @unittest.skipUnless(shutil.which('node'), 'Node is required for the actual review-script contract')
    def test_review_requires_blind_observation_before_assessment(self):
        self.fixture()
        out = self.root / 'results/blinding'
        build_pack(self.root, out, max_words=1, per_parasha=1)
        page = (out / 'review.html').read_text()
        script = re.search(r'<script>(.*?)</script>', page, re.S).group(1)
        harness = r'''
const vm=require('node:vm'),assert=require('node:assert/strict'),fs=require('node:fs');
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,{value:'',hidden:true,disabled:false,textContent:''});return elements.get(id)};
const saved=new Map(),context=vm.createContext({document:{getElementById:get},localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},Date});
vm.runInContext(fs.readFileSync(0,'utf8'),context);
assert.equal(get('expected').hidden,true);assert.equal(get('assessment').hidden,true);
get('reveal').onclick();assert.equal(get('expected').hidden,true);
get('reviewer').value='synthetic-reviewer';get('performed').value='unclear';get('transcript').value='synthetic observation';
get('reveal').onclick();assert.equal(get('expected').hidden,false);assert.equal(get('assessment').hidden,false);
assert.equal(get('performed').disabled,true);assert.equal(get('transcript').disabled,true);
assert.equal(vm.runInContext('Object.values(store)[0].annotation_status',context),'blind_phase_locked');
get('save').onclick();assert.equal(vm.runInContext('Object.values(store)[0].annotation_status',context),'blind_phase_locked');
get('correct').value='unclear';get('confidence').value='נמוך';get('save').onclick();
assert.equal(vm.runInContext('Object.values(store)[0].annotation_status',context),'human_reviewed');
assert.equal(vm.runInContext('Object.values(store)[0].human_correct',context),null);
assert.equal(vm.runInContext('Object.values(store)[0].blind_performed_taam',context),'unclear');
'''
        checked = subprocess.run(['node', '-e', harness], input=script, text=True, capture_output=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
