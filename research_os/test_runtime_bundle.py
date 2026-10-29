"""Bundle validation tests use tiny fixtures and never load ML models."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from webapp.runtime_bundle import (REQUIRED_ROLES, RuntimeBundleError, STATUS,
    bundle_digest,file_sha256,library_versions,load_bundle,public_summary,validate_bundle)
from research_os.model_bundle import checkpoint_class_order,discover


class RuntimeBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        files=[]
        for role in sorted(REQUIRED_ROLES):
            path=self.root/(role+'.fixture');path.write_text(role)
            files.append({'role':role,'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size})
        self.payload={'schema_version':1,'kind':'taamify.runtime-bundle','status':STATUS,
            'code_commit':'a'*40,'dirty_diff_sha256':'b'*64,'dirty_status_sha256':'c'*64,
            'binding_evidence':'observed_files_only','files':files,
            'runtime_options':{'channel_calibration':None},
            'library_versions':library_versions(),'frontend':{'SR':16000}}
        self.sign()

    def sign(self):
        self.payload['bundle_sha256']=bundle_digest(self.payload)

    def test_valid_manifest_binds_all_roles_without_importing_torch(self):
        with patch('builtins.__import__',wraps=__import__) as imports:
            self.assertEqual(validate_bundle(self.payload,self.root),self.payload)
        self.assertFalse(any(c.args[0]=='torch' for c in imports.call_args_list))

    def test_bundle_mutation_fails(self):
        self.payload['frontend']['SR']=22050
        with self.assertRaisesRegex(RuntimeBundleError,'digest mismatch'):
            validate_bundle(self.payload,self.root)

    def test_file_mutation_same_size_fails(self):
        path=self.root/'cnn.fixture';path.write_text('CNN')
        with self.assertRaisesRegex(RuntimeBundleError,'bytes changed'):
            validate_bundle(self.payload,self.root)

    def test_missing_file_fails(self):
        (self.root/'ctc_weights.fixture').unlink()
        with self.assertRaisesRegex(RuntimeBundleError,'missing'):
            validate_bundle(self.payload,self.root)

    def test_required_role_cannot_be_omitted_even_if_resigned(self):
        self.payload['files']=[e for e in self.payload['files'] if e['role']!='sequence'];self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'missing required role'):
            validate_bundle(self.payload,self.root)

    def test_duplicate_role_fails(self):
        self.payload['files'].append(copy.deepcopy(self.payload['files'][0]));self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'duplicate role'):
            validate_bundle(self.payload,self.root)

    def test_unknown_role_cannot_leak_a_path_in_the_public_role_list(self):
        self.payload['files'][0]['role']='/private/path';self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'unsupported role'):
            validate_bundle(self.payload,self.root)

    def test_commit_and_source_provenance_required(self):
        for name in ('code_commit','dirty_diff_sha256','dirty_status_sha256'):
            with self.subTest(name=name):
                data=copy.deepcopy(self.payload);data.pop(name);data['bundle_sha256']=bundle_digest(data)
                with self.assertRaises(RuntimeBundleError):validate_bundle(data,self.root)

    def test_traversal_absolute_and_noncanonical_paths_fail(self):
        for value in ['../outside','/etc/passwd','./cnn.fixture','foo//bar','foo\\bar','bad\0name']:
            with self.subTest(value=value):
                item=copy.deepcopy(self.payload);item['files'][0]['path']=value
                item['bundle_sha256']=bundle_digest(item)
                with self.assertRaises(RuntimeBundleError):
                    validate_bundle(item,self.root)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target=Path(outside)/'external';target.write_text('external')
            (self.root/'escape').symlink_to(target)
            self.payload['files'][0].update(path='escape',sha256=file_sha256(target),bytes=target.stat().st_size)
            self.sign()
            with self.assertRaisesRegex(RuntimeBundleError,'symlink escapes'):
                validate_bundle(self.payload,self.root)

    def test_class_order_disagreement_fails(self):
        for entry in self.payload['files']:
            if entry['role']=='cnn':entry['class_order']=['Munach','Mercha']
            if entry['role']=='verify_head':entry['class_order']=['Mercha','Munach']
        self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'class order differs'):
            validate_bundle(self.payload,self.root)

    def test_runtime_option_change_fails(self):
        with self.assertRaisesRegex(RuntimeBundleError,'runtime options'):
            validate_bundle(self.payload,self.root,{'channel_calibration':'policy.json'})

    def test_selected_channel_policy_must_be_bound(self):
        options={'channel_calibration':'policy.json'}
        self.payload['runtime_options']=options;self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'not bound'):
            validate_bundle(self.payload,self.root,options)
        path=self.root/'policy.json';path.write_text('{}')
        self.payload['files'].append({'role':'calibration_channel_candidate','path':'policy.json',
            'sha256':file_sha256(path),'bytes':2});self.sign()
        validate_bundle(self.payload,self.root,options)

    def test_unselected_candidate_cannot_sneak_into_bundle(self):
        path=self.root/'policy.json';path.write_text('{}')
        self.payload['files'].append({'role':'calibration_channel_candidate','path':'policy.json',
            'sha256':file_sha256(path),'bytes':2});self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'not selected'):
            validate_bundle(self.payload,self.root)

    def test_library_version_drift_is_rejected(self):
        self.payload['library_versions']['torch']='changed';self.sign()
        with self.assertRaisesRegex(RuntimeBundleError,'library versions changed'):
            validate_bundle(self.payload,self.root)

    def test_schema_and_training_provenance_overclaim_are_rejected(self):
        for key,value in [('schema_version',2),('kind','other'),('status','proven'),('binding_evidence','training_verified')]:
            with self.subTest(key=key):
                item=copy.deepcopy(self.payload);item[key]=value;item['bundle_sha256']=bundle_digest(item)
                with self.assertRaises(RuntimeBundleError):validate_bundle(item,self.root)

    def test_public_summary_contains_no_private_paths(self):
        result=public_summary(self.payload)
        encoded=json.dumps(result)
        self.assertNotIn(str(self.root),encoded)
        self.assertNotIn('fixture',encoded)
        self.assertEqual(set(result),{'schema_version','bundle_sha256','status','binding_evidence','roles'})
        self.assertEqual(public_summary(None)['status'],'unbound')

    def test_load_rejects_malformed_or_missing_selected_bundle(self):
        path=self.root/'bundle.json'
        with self.assertRaises(RuntimeBundleError):load_bundle(path,self.root)
        path.write_text('{')
        with self.assertRaises(RuntimeBundleError):load_bundle(path,self.root)
        path.write_text(json.dumps(self.payload))
        self.assertEqual(load_bundle(path,self.root)['bundle_sha256'],self.payload['bundle_sha256'])

    def test_literal_class_names_can_be_read_without_unpickle(self):
        path=self.root/'checkpoint.pth'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('checkpoint/data.pkl',pickle.dumps({'class_names':['Munach','Mercha']},protocol=2))
        with patch('pickle.loads',side_effect=AssertionError('unpickle attempted')):
            self.assertEqual(checkpoint_class_order(path),['Munach','Mercha'])

    def test_nonliteral_pickle_metadata_is_not_executed(self):
        class Unsafe:
            def __reduce__(self):return (print,('DO NOT EXECUTE',))
        path=self.root/'checkpoint.pth'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('checkpoint/data.pkl',pickle.dumps({'class_names':[Unsafe()]},protocol=2))
        with patch('builtins.print',side_effect=AssertionError('pickle code executed')):
            self.assertIsNone(checkpoint_class_order(path))

    def test_active_layout_discovery_reads_current_source_without_server_import(self):
        root=Path(__file__).resolve().parent.parent
        paths,frontend,options=discover(root)
        self.assertTrue(REQUIRED_ROLES <= paths.keys())
        self.assertEqual(frontend['SR'],16000)
        self.assertEqual(options,{'channel_calibration':None})


if __name__=='__main__':
    unittest.main()
