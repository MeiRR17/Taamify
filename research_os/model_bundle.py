"""Create/validate an observed serving bundle without loading any models.

Reads active paths and frontend values from Python AST, hashes local files,
and safely inspects literal checkpoint metadata with pickletools (no unpickle).
Writes a new ignored result file only; never downloads or swaps a checkpoint.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickletools
import subprocess
import zipfile

from webapp.runtime_bundle import (RuntimeBundleError, SCHEMA_VERSION, STATUS,
    bundle_digest, file_sha256, library_versions, load_bundle, public_summary, relative_file)

ROOT = Path(__file__).resolve().parent.parent


def eval_path(node, root):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == 'PROJECT_DIR':
        return root
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return Path(eval_path(node.left,root)) / eval_path(node.right,root)
    raise RuntimeBundleError('active runtime path is not a safe literal; update discovery explicitly')


def assignment(tree, name, function=None):
    body = tree.body
    if function:
        node = next((n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name == function),None)
        if node is None:
            raise RuntimeBundleError(f'active runtime function {function} is missing')
        nodes = ast.walk(node)
    else:
        nodes = iter(body)
    for node in nodes:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise RuntimeBundleError(f'active runtime value {name} is missing')


def relative(root, path):
    try:
        value = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeBundleError('runtime asset outside project root') from exc
    relative_file(root,value)
    return value


def checkpoint_class_order(path):
    """Read a literal string list only. Never invoke pickle's REDUCE/GLOBAL.

    NumPy-scalar lists in older files are intentionally left unknown rather
    than executing pickle or pretending a sidecar is checkpoint provenance.
    """
    if path.suffix != '.pth':
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            info = next((i for i in archive.infolist() if i.filename.endswith('/data.pkl')),None)
            if info is None or info.file_size > 16*1024*1024:
                return None
            ops = list(pickletools.genops(archive.read(info)))
        for index,(op,arg,pos) in enumerate(ops):
            if op.name not in {'BINUNICODE','SHORT_BINUNICODE'} or arg != 'class_names':
                continue
            names, saw_list = [], False
            for following,value,_ in ops[index+1:index+512]:
                if following.name in {'BINPUT','LONG_BINPUT','MEMOIZE','MARK'}:
                    continue
                if following.name == 'EMPTY_LIST' and not saw_list:
                    saw_list = True
                    continue
                if following.name in {'BINUNICODE','SHORT_BINUNICODE'} and saw_list:
                    names.append(value)
                    continue
                if following.name in {'APPENDS','APPEND'} and names:
                    return names if len(names) == len(set(names)) else None
                return None
    except (OSError,ValueError,zipfile.BadZipFile):
        return None
    return None


def discover(root, channel_policy=None):
    """Observe the current Torah/phrase-serving layout from source, not import."""
    root=Path(root).resolve()
    server=ast.parse((root/'webapp/server.py').read_text(encoding='utf-8'))
    ctc=ast.parse((root/'Data_Prep/ctc_align.py').read_text(encoding='utf-8'))
    paths={
        'cnn':eval_path(assignment(server,'ckpt_path','get_model'),root),
        'verify_head':eval_path(assignment(server,'VERIFY_HEAD_PATH'),root),
        'sequence':eval_path(assignment(server,'SEQUENCE_TAGGER_PATH'),root),
        'calibration_global':eval_path(assignment(server,'CALIBRATION_PATH'),root),
        'calibration_per_class':eval_path(assignment(server,'path','get_per_class_thresholds'),root)}
    channels=ast.literal_eval(assignment(server,'CHANNELS'))
    for name,cfg in channels.items():
        found=list((root/cfg['dir']).glob('*/*_best.pth'))
        if len(found) != 1:
            raise RuntimeBundleError(f'channel {name} has {len(found)} checkpoints; pin a unique artifact first')
        paths['channel_'+name]=found[0]
    local=eval_path(assignment(ctc,'LOCAL_MODEL'),root)
    for role,name in {'weights':'model.safetensors','config':'config.json',
                      'vocab':'vocab.json','preprocessor':'preprocessor_config.json',
                      'tokenizer':'tokenizer_config.json','special_tokens':'special_tokens_map.json'}.items():
        paths['ctc_'+role]=local/name
    if (local/'added_tokens.json').is_file():
        paths['ctc_added_tokens']=local/'added_tokens.json'
    sources={
        'frontend_server':'webapp/server.py', 'frontend_training':'10_train_pipeline.py',
        'frontend_verify':'15_binary_verify_model.py', 'frontend_sequence':'14_sequence_model.py',
        'frontend_alignment':'Data_Prep/verse_align.py', 'frontend_ctc':'Data_Prep/ctc_align.py',
        'frontend_text_normalization':'Data_Prep/clean_align_v2.py',
        'frontend_stress':'Data_Prep/stressed_syllable.py', 'frontend_sheva':'Data_Prep/sheva.py',
        'frontend_severity':'Data_Prep/severity.py',
        'frontend_channel_policy':'webapp/channel_calibration.py',
        'frontend_capture':'webapp/reading_contract.py', 'frontend_practice':'webapp/static/practice.html',
        'runtime_validator':'webapp/runtime_bundle.py'}
    paths.update({role:root/path for role,path in sources.items()})
    options={'channel_calibration':None}
    if channel_policy:
        path=Path(channel_policy)
        path=path if path.is_absolute() else root/path
        options['channel_calibration']=relative(root,path)
        paths['calibration_channel_candidate']=path
    frontend={name:ast.literal_eval(assignment(server,name)) for name in
        ('SR','N_MELS','FMAX','MAX_FRAMES','PAD_DB','CONTEXT_WEIGHT','ALIGN_GATE_SCORE',
         'CHANNEL_WINDOW','CORPUS_WINDOW_S','CORPUS_LEAD_IN_S','SHIFT_CANDIDATES','TTA_OFFSETS')}
    frontend.update({'mel_n_fft':2048,'mel_hop_length':512,'mel_center':True,
        'normalization':'per_word_zscore','description':'observed source values, not training provenance'})
    return {role:relative(root,path) for role,path in paths.items()}, frontend, options


def create_bundle(root, channel_policy=None):
    root=Path(root).resolve()
    paths,frontend,options=discover(root,channel_policy)
    files=[]
    for role,name in sorted(paths.items()):
        path=relative_file(root,name)
        entry={'role':role,'path':name,'sha256':file_sha256(path),'bytes':path.stat().st_size}
        names=checkpoint_class_order(path)
        if names is not None:
            entry['class_order']=names
            entry['class_order_evidence']='checkpoint_literal_metadata_without_unpickle'
        elif path.suffix == '.pth':
            entry['class_order_evidence']='not_safely_readable_without_unpickle'
        files.append(entry)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root)
    import hashlib
    payload={'schema_version':SCHEMA_VERSION,'kind':'taamify.runtime-bundle',
        'status':STATUS,'binding_evidence':'observed_files_only',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'code_commit':git('rev-parse','HEAD').decode().strip(),
        'dirty_diff_sha256':hashlib.sha256(git('diff','--binary','HEAD')).hexdigest(),
        'dirty_status_sha256':hashlib.sha256(git('status','--porcelain','-z')).hexdigest(),
        'runtime_options':options,'frontend':frontend,'library_versions':library_versions(),'files':files,
        'limitations':['File identity is observed now; legacy training pairings are not retroactively proven.',
            'Selected files are checked at startup; external model pretraining data are not certified.',
            'No claim of learner-error accuracy, new-reader generalization, or supported haftara runtime.']}
    payload['bundle_sha256']=bundle_digest(payload)
    return payload


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create');create.add_argument('--out',type=Path,required=True)
    create.add_argument('--channel-policy',type=Path)
    validate=sub.add_parser('validate');validate.add_argument('--bundle',type=Path,required=True)
    validate.add_argument('--channel-policy',type=Path)
    args=ap.parse_args(argv)
    try:
        channel=args.channel_policy or os.environ.get('TAAMIFY_CHANNEL_CALIBRATION')
        options={'channel_calibration':relative(ROOT,ROOT/channel) if channel else None}
        if args.command=='create':
            out=args.out if args.out.is_absolute() else ROOT/args.out
            allowed=ROOT/'results/research_os/runtime'
            try:
                out.resolve().relative_to(allowed.resolve())
            except ValueError as exc:
                raise RuntimeBundleError('bundle output must stay under ignored results/research_os/runtime/') from exc
            if out.exists():
                raise RuntimeBundleError('bundle output already exists; preserve previous evidence')
            payload=create_bundle(ROOT,channel)
            from webapp.runtime_bundle import validate_bundle
            validate_bundle(payload,ROOT,options)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x',encoding='utf-8') as stream:
                json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
        else:
            payload=load_bundle(args.bundle,ROOT,options)
        print(json.dumps(public_summary(payload),ensure_ascii=False,indent=2))
    except (RuntimeBundleError,OSError,ValueError,TypeError,subprocess.CalledProcessError) as exc:
        ap.exit(2,f'runtime bundle refused: {exc}\n')


if __name__=='__main__':
    main()
