"""Bounded fresh DeepCNN development pilot; never evaluates test/calibration.

Uses only the new source-independent train/development arrays. Other artifacts
are read as opaque bytes for integrity hashes only. All outputs stay in a NEW
local candidate directory; this does not change the serving model or old trainer.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import time

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn

from research_os.prepare_independent_dataset import (
    DataContractError, SPLITS, overlap_audit, sha256_file, stable_hash)

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / '10_train_pipeline.py'


def verify_training_inputs(data_dir):
    """Verify hashes without decoding test or calibration arrays/manifests."""
    data_dir = Path(data_dir)
    if (data_dir/'BUILDING').exists() or (data_dir/'BUILD_FAILED').exists():
        raise DataContractError('dataset build is incomplete')
    metadata = json.loads((data_dir/'metadata.json').read_text())
    if metadata.get('status') != 'ready' or metadata.get('split') != 'source_connected_components':
        raise DataContractError('requires a ready independent dataset')
    for name, expected in metadata['file_sha256'].items():
        if Path(name).name != name or sha256_file(data_dir/name) != expected:
            raise DataContractError(f'artifact hash mismatch: {name}')
    fingerprint = stable_hash({k:metadata[k] for k in (
        'feature_contract','seed','heldout_parashot','fractions_of_non_test',
        'provenance','manifest_assignment_sha256')})
    if metadata['dataset_fingerprint'] != fingerprint:
        raise DataContractError('dataset fingerprint mismatch')
    arrays, records, assignments = {}, [], {}
    for split in ('train','val'):
        rows=[json.loads(s) for s in (data_dir/f'{split}_manifest.jsonl').read_text().splitlines()]
        with np.load(data_dir/f'{split}.npz') as data:
            x,y,ids,sources=data['X'],data['y'],data['sample_ids'],data['source_rows']
        if len(rows) != metadata[split+'_samples'] or len(x)!=len(rows) or len(y)!=len(rows):
            raise DataContractError('manifest/array length mismatch')
        if x.shape[1:] != tuple(metadata['input_shape']) or not np.isfinite(x).all():
            raise DataContractError('feature shape/nonfinite values violate contract')
        for row,sid,source,label in zip(rows,ids,sources,y):
            if row['sample_id']!=str(sid) or row['source_row']!=int(source) or row['split']!=split or \
                    row['label']!=metadata['class_names'][int(label)]:
                raise DataContractError('row identity/label mismatch')
            if str(sid) in assignments:
                raise DataContractError('duplicate row ID')
            assignments[str(sid)]=split
            records.append(row)
        arrays[split]=(torch.from_numpy(x),torch.from_numpy(y))
    overlap_audit(records,assignments)
    train_labels=arrays['train'][1].numpy()
    counts=np.bincount(train_labels,minlength=len(metadata['class_names']))
    if (counts==0).any():
        raise DataContractError('training class has zero support')
    return metadata,arrays


def validate_resume(checkpoint, binding):
    if checkpoint.get('status')!='development_pilot' or checkpoint.get('binding')!=binding:
        raise DataContractError('resume checkpoint does not match dataset, recipe, classes or code')


def epoch_batches(size,batch_size,seed,epoch):
    generator=torch.Generator().manual_seed(seed+epoch)
    indices=torch.randperm(size,generator=generator)
    batches=list(indices.split(batch_size))
    if len(batches)>1 and len(batches[-1])==1:
        batches[-2]=torch.cat([batches[-2],batches[-1]])
        batches.pop()
    return batches


def development_metrics(labels,predictions,class_names):
    indices=list(range(len(class_names)))
    precision,recall,f1,support=precision_recall_fscore_support(
        labels,predictions,labels=indices,zero_division=0)
    return {'accuracy':float(np.mean(np.asarray(labels)==np.asarray(predictions))),
            'macro_f1':float(np.mean(f1)),
            'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),
                'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(class_names)},
            'confusion_matrix':confusion_matrix(labels,predictions,labels=indices).tolist()}


def snapshot_rng(device):
    result={'torch_cpu':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
    if device.type=='mps' and hasattr(torch.mps,'get_rng_state'):
        result['torch_mps']=torch.mps.get_rng_state()
    return result


def restore_rng(state,device):
    torch.set_rng_state(state['torch_cpu'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if device.type=='mps' and 'torch_mps' in state:
        torch.mps.set_rng_state(state['torch_mps'])


def save_atomic(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def synchronize(device):
    if device.type=='mps':
        torch.mps.synchronize()


def train(data_dir,out,seed=20261028,max_epochs=5,max_seconds=600,batch_size=64,
          learning_rate=.001,weight_decay=.0001,patience=3,device_name=None,resume=None):
    if not 1<=max_epochs<=5 or not 0<max_seconds<=600:
        raise DataContractError('pilot limited to at most five epochs and 600 total seconds')
    if batch_size<2 or patience<1:
        raise DataContractError('batch size/patience invalid')
    out=Path(out)
    if resume is None:
        if out.exists():
            raise FileExistsError('new run requires a new output directory')
    elif not out.exists() or Path(resume).resolve().parent!=out.resolve():
        raise DataContractError('resume must use a checkpoint from this run directory')
    invocation_start=time.monotonic()
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    metadata,arrays=verify_training_inputs(data_dir)
    verify_seconds=time.monotonic()-invocation_start
    recipe={'architecture':'deep_cnn','seed':seed,'loss':'balanced_cross_entropy',
            'augmentation':False,'optimizer':'Adam','learning_rate':learning_rate,
            'weight_decay':weight_decay,'batch_size':batch_size,'max_epochs':max_epochs,
            'max_seconds':max_seconds,'patience':patience,'threads':1,
            'selection':'development_macro_f1','initialization':'fresh'}
    binding={'dataset_fingerprint':metadata['dataset_fingerprint'],
             'dataset_metadata_sha256':sha256_file(Path(data_dir)/'metadata.json'),
             'class_names':metadata['class_names'],'recipe':recipe,
             'runner_sha256':sha256_file(__file__),'architecture_source_sha256':sha256_file(TRAINER)}
    device=torch.device(device_name or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    if device.type=='mps':
        torch.mps.manual_seed(seed)
    model_spec=importlib.util.spec_from_file_location('independent_architecture',TRAINER)
    architecture=importlib.util.module_from_spec(model_spec);model_spec.loader.exec_module(architecture)
    model=architecture.MODEL_REGISTRY['deep_cnn'](num_classes=len(metadata['class_names'])).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    counts=np.bincount(arrays['train'][1].numpy(),minlength=len(metadata['class_names']))
    weights=len(arrays['train'][1])/(len(counts)*counts)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    epoch=cursor=completed=stale=0
    epoch_loss=0.0;epoch_examples=0;best=-1.0;history=[];elapsed_before=0.0
    if resume:
        checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
        validate_resume(checkpoint,binding)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch,cursor,completed,stale=(checkpoint[k] for k in ('epoch_index','batch_cursor','completed_epochs','stale_epochs'))
        epoch_loss=checkpoint['epoch_loss_sum'];epoch_examples=checkpoint['epoch_examples']
        best=checkpoint['best_macro_f1'];history=checkpoint['history'];elapsed_before=checkpoint['elapsed_seconds']
        restore_rng(checkpoint['rng_state'],device)
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/'recipe.json').write_text(json.dumps({'status':'development_pilot','binding':binding},indent=2)+'\n')
        (out/'executed_runner.py').write_bytes(Path(__file__).read_bytes())
    deadline=invocation_start+max_seconds-elapsed_before
    stop_reason='max_epochs'

    def checkpoint_payload():
        synchronize(device)
        return {'status':'development_pilot','binding':binding,
            'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
            'class_names':metadata['class_names'],'architecture':'deep_cnn',
            'dataset_fingerprint':metadata['dataset_fingerprint'],'epoch_index':epoch,
            'batch_cursor':cursor,'completed_epochs':completed,'stale_epochs':stale,
            'epoch_loss_sum':epoch_loss,'epoch_examples':epoch_examples,
            'best_macro_f1':best,'history':history,'rng_state':snapshot_rng(device),
            'elapsed_seconds':elapsed_before+time.monotonic()-invocation_start}

    for current_epoch in range(epoch,max_epochs):
        epoch=current_epoch
        epoch_started=time.monotonic()
        batches=epoch_batches(len(arrays['train'][1]),batch_size,seed,epoch)
        model.train()
        for batch_index in range(cursor,len(batches)):
            if time.monotonic()>=deadline:
                stop_reason='time_budget';break
            indices=batches[batch_index]
            x=arrays['train'][0][indices].to(device)
            y=arrays['train'][1][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss):
                raise DataContractError('nonfinite training loss')
            loss.backward();optimizer.step()
            epoch_loss+=float(loss.detach().cpu())*len(y)
            epoch_examples+=len(y);cursor=batch_index+1
        synchronize(device)
        train_seconds=time.monotonic()-epoch_started
        if cursor<len(batches):
            break
        model.eval();val_started=time.monotonic();predictions=[];labels=[]
        with torch.inference_mode():
            for lo in range(0,len(arrays['val'][1]),batch_size):
                if time.monotonic()>=deadline:
                    stop_reason='time_budget_validation_incomplete';break
                x=arrays['val'][0][lo:lo+batch_size].to(device)
                predictions.extend(model(x).argmax(dim=1).cpu().tolist())
                labels.extend(arrays['val'][1][lo:lo+batch_size].tolist())
        synchronize(device)
        if len(labels)!=len(arrays['val'][1]):
            break
        metrics=development_metrics(labels,predictions,metadata['class_names'])
        completed=epoch+1
        entry={'epoch':completed,'train_examples':epoch_examples,
               'train_loss':epoch_loss/epoch_examples,'train_seconds':round(train_seconds,3),
               'validation_seconds':round(time.monotonic()-val_started,3),'validation':metrics}
        history.append(entry)
        improved=metrics['macro_f1']>best+1e-8
        best=max(best,metrics['macro_f1']);stale=0 if improved else stale+1
        epoch=completed;cursor=0;epoch_loss=0.0;epoch_examples=0
        payload=checkpoint_payload()
        save_atomic(out/'last.pth',payload)
        if improved:
            save_atomic(out/'best.pth',payload)
        with (out/'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
        print(json.dumps({'epoch':completed,'macro_f1':metrics['macro_f1'],
            'accuracy':metrics['accuracy'],'train_seconds':entry['train_seconds'],
            'validation_seconds':entry['validation_seconds']}),flush=True)
        if stale>=patience:
            stop_reason='early_stopping';break
    save_atomic(out/'last.pth',checkpoint_payload())
    synchronize(device)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes=int(rss if platform.system()=='Darwin' else rss*1024)
    report={'status':'development_pilot','stop_reason':stop_reason,
            'binding':binding,'completed_epochs':completed,'current_epoch_index':epoch,
            'current_batch_cursor':cursor,'best_macro_f1':best if best>=0 else None,
            'history':history,'device':str(device),'torch_version':torch.__version__,
            'peak_process_rss_bytes':rss_bytes,'verification_seconds':round(verify_seconds,3),
            'elapsed_seconds':round(elapsed_before+time.monotonic()-invocation_start,3),
            'best_checkpoint_sha256':sha256_file(out/'best.pth') if (out/'best.pth').exists() else None,
            'last_checkpoint_sha256':sha256_file(out/'last.pth'),
            'numpy_decoded_splits':['train','val'],'test_or_calibration_evaluated':False,
            'other_artifacts_access':'opaque SHA-256 integrity bytes only; no test/calibration decoding',
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'limitations':metadata['limitations']+[
                'Short fresh-initialization pilot; not a production-quality model or accuracy improvement claim.',
                'Old checkpoints may have trained on this development split historically; not a fair unseen baseline.',
                'MPS deterministic algorithms are requested with warnings; bitwise cross-device reproducibility is not guaranteed.',
                'Budget checked between batches; final checkpoint/IO may add a small bounded shutdown overhead.']}
    (out/'pilot_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--max-epochs',type=int,default=5)
    parser.add_argument('--max-seconds',type=float,default=600)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--device',choices=['cpu','mps'])
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    report=train(args.data_dir,args.out,args.seed,args.max_epochs,args.max_seconds,
                 args.batch_size,device_name=args.device,resume=args.resume)
    print(json.dumps({k:report[k] for k in ('status','stop_reason','completed_epochs',
        'best_macro_f1','elapsed_seconds','peak_process_rss_bytes','test_or_calibration_evaluated')},indent=2))


if __name__=='__main__':
    main()
