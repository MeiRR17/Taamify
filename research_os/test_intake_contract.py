"""Meaningful intake regression tests with synthetic tones, never real voices."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from Data_Prep import intake_contract as contract
from Data_Prep import voice_intake
from Data_Prep import ingest_collected_phrases as ingest

ROOT = Path(__file__).resolve().parent.parent


class IntakeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.wav = self.path/'test-clip.wav'
        with wave.open(str(self.wav), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(struct.pack('<1600h', *([123]*1600)))
        self.record = {
            'schema_version':2, 'clip_id':'test-clip', 'consent':True,
            'consent_version':contract.CONSENT_VERSION,
            'consent_scope':['local_research','training'],
            'voice_id':'voice_test', 'session_id':'session_test',
            'tradition':contract.TRADITION, 'source_root_id':'capture:test-clip',
            'audio_sha256':contract.file_digest(self.wav),
            'words':['אָב'], 'taamim':['Munach'],
            'statuses':['ok'], 'aligned':True, 'align_score':-1.0}
        self.annotation = {
            'status':'approved', 'label_source':'human-reviewed',
            'reviewed_by':'reviewer_test', 'reviewed_at':'2026-10-28T10:00:00+00:00',
            'reading_quality':'clear', 'audio_sha256':self.record['audio_sha256'],
            'record_sha256':contract.canonical_digest(self.record),
            'performed_words':['אָב'], 'performed_taamim':['Tipecha']}

    def approval(self):
        payload = voice_intake.build_approval([self.record], {'test-clip':self.annotation}, self.path)
        path = self.path/'training_safe.json'
        path.write_text(json.dumps(payload))
        return path

    def write_inputs(self):
        (self.path/'phrases_log.jsonl').write_text(json.dumps(self.record)+'\n')
        (self.path/'intake_decisions.json').write_text(json.dumps({
            'schema_version':2, 'clips':{'test-clip':self.annotation}}))

    def validate(self, path):
        return contract.validate_approval(path, [self.record], {'test-clip':self.annotation}, self.path)

    def test_expected_taam_is_never_the_observed_training_label(self):
        path = self.approval()
        approved = self.validate(path)
        self.assertEqual(approved[0][2]['performed_taamim'], ['Tipecha'])
        self.assertEqual(self.record['taamim'], ['Munach'])
        out = self.path/'new-features'
        observed_words = []
        def align(audio, words):
            observed_words.extend(w['raw'] for w in words)
            return {0:(0.0, 0.1)}, -1.0, .02
        tools = (lambda p:np.zeros(1600), align, lambda w:w,
                 lambda y,s,e:np.arange(128*87,dtype=np.float32).reshape(128,87))
        summary = ingest.create_snapshot(approved,self.path,out,path,-2.5,tools)
        self.assertEqual(summary['written_words'],1)
        meta = json.loads(next(out.glob('*_meta.json')).read_text())
        self.assertEqual(meta[0]['taam'],'Tipecha')
        self.assertEqual(meta[0]['taam_expected'],'Munach')
        self.assertEqual(meta[0]['annotation_sha256'],contract.canonical_digest(self.annotation))
        self.assertEqual(observed_words,['אָב'])

    def test_model_agreement_without_human_observation_stays_quarantined(self):
        tier, reason = voice_intake.classify_phrase(self.record,.9999,{})
        self.assertEqual(tier,'quarantined')
        self.assertEqual(reason,'human_annotation_missing')
        payload = voice_intake.build_approval([self.record],{},self.path)
        self.assertEqual(payload['entries'],[])

    def test_explicit_consent_and_identity_required_before_human_override(self):
        cases = [('consent','true'),('consent_version','unknown'),
                 ('consent_scope',['local_research']),('voice_id','anon'),
                 ('session_id',''),('tradition','ashkenazi'),('source_root_id','')]
        for field,value in cases:
            with self.subTest(field=field):
                rec=copy.deepcopy(self.record);rec[field]=value
                tier,_=voice_intake.classify_phrase(rec,None,{'test-clip':self.annotation})
                self.assertEqual(tier,'rejected')

    def test_path_like_clip_identifier_is_rejected(self):
        self.record['clip_id']='../../outside'
        self.assertEqual(contract.capture_problem(self.record),'clip_id_invalid')

    def test_legacy_gold_string_and_claimed_tier_do_not_approve(self):
        self.record['tier']='gold'
        tier,_=voice_intake.classify_phrase(self.record,None,{'test-clip':'gold'})
        self.assertEqual(tier,'quarantined')

    def test_pseudo_labels_and_unclear_audio_cannot_be_human_gold(self):
        for field,value in [('label_source','pseudo-label'),('reading_quality','unclear'),
                            ('status','pending'),('reviewed_by',None),('reviewed_at','2026-10-28')]:
            with self.subTest(field=field):
                annotation=copy.deepcopy(self.annotation);annotation[field]=value
                entry,problem=contract.review_entry(self.record,annotation,self.path)
                self.assertIsNone(entry)
                self.assertIsNotNone(problem)

    def test_changed_word_is_quarantined_for_separate_error_alignment(self):
        self.annotation['performed_words']=['אֵם']
        entry,problem=contract.review_entry(self.record,self.annotation,self.path)
        self.assertIsNone(entry)
        self.assertEqual(problem,'performed_words_differ_requires_error_dataset')

    def test_invalid_performed_label_shape_or_value_rejected(self):
        for value in [[],['unknown'],['Tipecha','Munach'],[{'bad':True}],[None]]:
            with self.subTest(value=value):
                annotation=copy.deepcopy(self.annotation);annotation['performed_taamim']=value
                entry,_=contract.review_entry(self.record,annotation,self.path)
                self.assertIsNone(entry)

    def test_changed_audio_invalidates_approval(self):
        path=self.approval()
        self.wav.write_bytes(self.wav.read_bytes()+b'changed')
        with self.assertRaisesRegex(contract.IntakeError,'audio_bytes_changed'):
            self.validate(path)

    def test_changed_record_invalidates_approval(self):
        path=self.approval()
        self.record['session_id']='different_session'
        with self.assertRaisesRegex(contract.IntakeError,'record_hash_mismatch'):
            self.validate(path)

    def test_changed_human_label_invalidates_approval(self):
        path=self.approval()
        self.annotation['performed_taamim']=['Mercha']
        with self.assertRaisesRegex(contract.IntakeError,'provenance or performed labels changed'):
            self.validate(path)

    def test_changed_manifest_invalidates_digest(self):
        path=self.approval()
        data=json.loads(path.read_text());data['entries']=[];path.write_text(json.dumps(data))
        with self.assertRaisesRegex(contract.IntakeError,'digest mismatch'):
            self.validate(path)

    def test_duplicate_audio_not_promoted_twice(self):
        other=copy.deepcopy(self.record);other['clip_id']='other-clip'
        (self.path/'other-clip.wav').write_bytes(self.wav.read_bytes())
        annotation=copy.deepcopy(self.annotation);annotation['record_sha256']=contract.canonical_digest(other)
        payload=voice_intake.build_approval([self.record,other],
            {'test-clip':self.annotation,'other-clip':annotation},self.path)
        self.assertEqual(len(payload['entries']),1)
        self.assertEqual(payload['rejected_reasons']['duplicate_audio_bytes'],1)

    def test_duplicate_capture_log_ids_fail_instead_of_overwriting(self):
        log=self.path/'phrases_log.jsonl'
        log.write_text((json.dumps(self.record)+'\n')*2)
        with self.assertRaisesRegex(contract.IntakeError,'duplicate clip_id'):
            contract.load_records(log)

    def test_cli_missing_approval_fails_before_any_training_output(self):
        self.write_inputs()
        before={p.name:p.read_bytes() for p in self.path.iterdir()}
        out=self.path/'forbidden-output'
        proc=subprocess.run([sys.executable,'Data_Prep/ingest_collected_phrases.py',
            '--phrases-dir',str(self.path),'--out-dir',str(out)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(proc.returncode,2)
        self.assertIn('approval manifest missing',proc.stderr)
        self.assertFalse(out.exists())
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.path.iterdir()})

    def test_cli_old_allowed_ids_do_not_approve_training(self):
        self.write_inputs()
        (self.path/'training_safe.json').write_text(json.dumps({'allowed_clip_ids':['test-clip']}))
        out=self.path/'forbidden-output'
        proc=subprocess.run([sys.executable,'Data_Prep/ingest_collected_phrases.py',
            '--phrases-dir',str(self.path),'--out-dir',str(out)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(proc.returncode,2)
        self.assertIn('invalid approval schema',proc.stderr)
        self.assertFalse(out.exists())

    def test_cli_validates_approved_inputs_without_model_loading(self):
        self.write_inputs();self.approval()
        proc=subprocess.run([sys.executable,'Data_Prep/ingest_collected_phrases.py',
            '--phrases-dir',str(self.path),'--validate-only'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['approved_clips'],1)

    def test_existing_feature_snapshot_is_preserved(self):
        path=self.approval();approved=self.validate(path)
        out=self.path/'existing';out.mkdir();(out/'keep').write_text('original')
        with self.assertRaisesRegex(contract.IntakeError,'directory exists'):
            ingest.create_snapshot(approved,self.path,out,path,-2.5)
        self.assertEqual((out/'keep').read_text(),'original')

    def test_bad_feature_leaves_no_partial_snapshot(self):
        path=self.approval();approved=self.validate(path)
        out=self.path/'bad'
        tools=(lambda p:np.zeros(1600),lambda y,w:({0:(0,.1)},-1,.02),lambda w:w,
               lambda y,s,e:np.full((128,87),np.nan))
        with self.assertRaisesRegex(contract.IntakeError,'invalid performed-word feature'):
            ingest.create_snapshot(approved,self.path,out,path,-2.5,tools)
        self.assertFalse(out.exists())

    def test_server_capture_passes_intake_but_has_no_human_training_label(self):
        from webapp import server
        from webapp import reading_contract
        capture=reading_contract.capture_metadata('true','voice_12345','session_12345',
            reading_contract.CONSENT_VERSION,json.dumps(reading_contract.CONSENT_SCOPES),
            reading_contract.TRADITION)
        collected=self.path/'server-captured'
        with patch.object(server,'PHRASES_DIR',collected), \
             patch.object(server,'PHRASES_LOG',collected/'phrases_log.jsonl'):
            clip_id=server.log_phrase(self.wav.read_bytes(),['אָב'],['Munach'],
                {'aligned':True,'align_score':-1.0,'words':[{'status':'ok'}]},capture)
        rows=contract.load_records(collected/'phrases_log.jsonl')
        self.assertEqual(rows[0]['clip_id'],clip_id)
        self.assertIsNone(contract.capture_problem(rows[0]))
        tier,reason=voice_intake.classify_phrase(rows[0],None,{})
        self.assertEqual((tier,reason),('quarantined','human_annotation_missing'))

    def test_cli_review_template_is_pending_and_never_copies_expected_labels(self):
        self.write_inputs()
        template=self.path/'pending-review.json'
        proc=subprocess.run([sys.executable,'Data_Prep/voice_intake.py',
            '--phrases-dir',str(self.path),'--review-template',str(template)],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        entry=json.loads(template.read_text())['clips']['test-clip']
        self.assertEqual(entry['status'],'pending')
        self.assertIsNone(entry['performed_words'])
        self.assertIsNone(entry['performed_taamim'])
        self.assertEqual(entry['record_sha256'],contract.canonical_digest(self.record))
        self.assertEqual(entry['audio_sha256'],contract.file_digest(self.wav))

    def test_legacy_file_spec_import_used_by_audit_still_works(self):
        spec=importlib.util.spec_from_file_location('intake_spec_audit',ROOT/'Data_Prep/voice_intake.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.classify_phrase({'clip_id':'audit'},None,{})[0],'rejected')


if __name__=='__main__':
    unittest.main()
