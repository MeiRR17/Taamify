"""Regression contracts for source-independent, immutable dataset preparation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_os.prepare_independent_dataset import (
    DataContractError, feature_hash, normalize_block, overlap_audit,
    partition_records, prepare, quality_reason, verify_dataset)


def record(i, label='A', parasha='1', verse=None, feature=None):
    return {'sample_id': str(i), 'label':label, 'parasha': parasha,
            'source_root_id': 'block:' + str(i),
            'verse_group_id': 'verse:' + str(verse if verse is not None else i),
            'feature_sha256': feature or 'feature:' + str(i)}


class IndependentDatasetTests(unittest.TestCase):
    def test_source_and_exact_duplicates_remain_together(self):
        records = [record(i, label='AB'[i%2]) for i in range(40)]
        records[1]['verse_group_id'] = records[0]['verse_group_id']
        records[4]['feature_sha256'] = records[0]['feature_sha256']
        assigned, quarantined = partition_records(records, ['A','B'])
        self.assertFalse(quarantined)
        self.assertEqual(assigned['0'], assigned['1'])
        self.assertEqual(assigned['0'], assigned['4'])
        self.assertFalse(any(any(v.values()) for v in overlap_audit(records,assigned).values()))

    def test_cross_test_duplicate_component_is_quarantined(self):
        records = [record(i) for i in range(30)] + [record(30,parasha='5',feature='feature:0')]
        assigned, quarantine = partition_records(records,['A'])
        self.assertEqual({r['sample_id'] for r in quarantine}, {'0','30'})
        self.assertNotIn('0',assigned)
        self.assertNotIn('30',assigned)

    def test_conflicting_labels_are_not_silently_trained(self):
        records = [record(i,label='AB'[i%2]) for i in range(40)]
        records[1]['feature_sha256'] = records[0]['feature_sha256']
        _, quarantine = partition_records(records,['A','B'])
        self.assertEqual({r['reason'] for r in quarantine}, {'conflicting_duplicate_labels'})

    def test_assignments_stable_even_if_input_order_changes(self):
        rows=[record(i,label='AB'[i%2],parasha='5' if i>40 else '1') for i in range(50)]
        first,_=partition_records(rows,['A','B'])
        second,_=partition_records(list(reversed(rows)),['A','B'])
        self.assertEqual(first,second)
        self.assertTrue(all(first[str(i)]=='test' for i in range(41,50)))

    def test_rare_class_reserved_for_training_without_splitting(self):
        rows=[record(i) for i in range(30)] + [record(40,label='rare')]
        assigned,_=partition_records(rows,['A','rare'])
        self.assertEqual(assigned['40'],'train')

    def test_no_non_test_support_for_declared_class_fails(self):
        with self.assertRaises(DataContractError):
            partition_records([record(1),record(2,label='B',parasha='5')],['A','B'])

    def test_invalid_ids_and_ratios_fail(self):
        with self.assertRaises(DataContractError):
            partition_records([record(1),record(1)],['A'])
        with self.assertRaises(DataContractError):
            partition_records([record(1)],['A'],val_fraction=.7,calibration_fraction=.4)

    def test_quality_and_normalization(self):
        silent=np.full((128,87),-80,dtype=np.float32)
        self.assertEqual(quality_reason(silent),'too_silent')
        x=np.random.default_rng(42).normal(-20,5,(3,128,87)).astype(np.float32)
        normalized=normalize_block(x)
        np.testing.assert_allclose(normalized.mean(axis=(1,2)),0,atol=1e-6)
        self.assertNotEqual(feature_hash(normalized[0]),feature_hash(normalized[1]))

    def test_real_files_include_ids_hashes_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rng=np.random.default_rng(12)
            x=rng.normal(-20,5,(80,128,87)).astype(np.float32)
            np.save(root/'x.npy',x)
            rows=[{'taam':'AB'[i%2],'verse_ref':f'1.1.{i}',
                   'segment_block':'1','parasha':'5' if i>=60 else '1',
                   'word_raw':'test','start':0,'end':1} for i in range(len(x))]
            (root/'meta.json').write_text(json.dumps(rows))
            (root/'classes.json').write_text(json.dumps({'class_names':['A','B']}))
            report=prepare(root/'x.npy',root/'meta.json',root/'classes.json',root/'new')
            self.assertEqual(report['status'],'ready')
            self.assertEqual(report['test_samples'],20)
            for split in ('train','val','calibration','test'):
                with np.load(root/'new'/f'{split}.npz') as data:
                    self.assertEqual(len(data['sample_ids']),len(data['y']))
                self.assertIn(split+'.npz',report['file_sha256'])
            self.assertFalse((root/'new/BUILDING').exists())
            self.assertEqual(verify_dataset(root/'new')['dataset_fingerprint'],report['dataset_fingerprint'])
            with self.assertRaises(FileExistsError):
                prepare(root/'x.npy',root/'meta.json',root/'classes.json',root/'new')
            manifest=root/'new/val_manifest.jsonl'
            manifest.write_text(manifest.read_text()+'{}\n')
            with self.assertRaises(DataContractError):
                verify_dataset(root/'new')


if __name__=='__main__':
    unittest.main()
