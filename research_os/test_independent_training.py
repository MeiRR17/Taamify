"""Pilot contracts: no test/calibration decoding, exact resume bindings, metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from research_os.prepare_independent_dataset import DataContractError, prepare
from research_os.train_independent_candidate import (
    development_metrics, epoch_batches, validate_resume, verify_training_inputs)


class IndependentTrainingTests(unittest.TestCase):
    def test_batch_order_is_resume_safe_and_covers_each_example_once(self):
        batches=epoch_batches(129,64,23,2)
        self.assertEqual([len(b) for b in batches],[64,65])
        self.assertEqual(sorted(torch.cat(batches).tolist()),list(range(129)))
        self.assertTrue(torch.equal(torch.cat(batches),torch.cat(epoch_batches(129,64,23,2))))
        self.assertFalse(torch.equal(torch.cat(batches),torch.cat(epoch_batches(129,64,23,3))))

    def test_metrics_keep_missing_classes_and_expose_precision(self):
        result=development_metrics([0,0,1],[1,1,1],['A','B','rare'])
        self.assertEqual(result['per_class']['rare']['support'],0)
        self.assertAlmostEqual(result['per_class']['B']['precision'],1/3)
        self.assertAlmostEqual(result['macro_f1'],1/6)

    def test_resume_requires_exact_dataset_classes_recipe_and_code(self):
        binding={'dataset_fingerprint':'data','class_names':['A','B'],
                 'recipe':{'seed':23},'runner_sha256':'runner'}
        checkpoint={'status':'development_pilot','binding':binding}
        validate_resume(checkpoint,binding)
        for key,value in [('dataset_fingerprint','other'),('class_names',['B','A']),
                          ('recipe',{'seed':24}),('runner_sha256','changed')]:
            with self.assertRaises(DataContractError):
                validate_resume(checkpoint,{**binding,key:value})
        with self.assertRaises(DataContractError):
            validate_resume({**checkpoint,'status':'production'},binding)

    def test_integrity_checks_never_decode_test_or_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            x=np.random.default_rng(3).normal(-20,5,(60,128,87)).astype(np.float32)
            np.save(root/'x.npy',x)
            meta=[{'taam':'AB'[i%2],'verse_ref':f'1.1.{i}',
                   'segment_block':'1','parasha':'5' if i>=40 else '1',
                   'word_raw':'test','start':0,'end':1} for i in range(60)]
            (root/'meta.json').write_text(json.dumps(meta))
            (root/'classes.json').write_text(json.dumps({'class_names':['A','B']}))
            prepare(root/'x.npy',root/'meta.json',root/'classes.json',root/'data')
            decoded=[]
            original=np.load
            def guarded(path,*args,**kwargs):
                decoded.append(Path(path).name)
                self.assertIn(Path(path).name,['train.npz','val.npz'])
                return original(path,*args,**kwargs)
            with patch('research_os.train_independent_candidate.np.load',side_effect=guarded):
                report,arrays=verify_training_inputs(root/'data')
            self.assertEqual(decoded,['train.npz','val.npz'])
            self.assertEqual(set(arrays),{'train','val'})
            self.assertEqual(report['status'],'ready')
            with (root/'data/train.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(DataContractError):
                verify_training_inputs(root/'data')


if __name__=='__main__':
    unittest.main()
