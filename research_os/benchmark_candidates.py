"""Local, paired canonical-recognition diagnostic on real contiguous phrases.

This is not an unseen-speaker or learner-error benchmark. Expected text goes
only to the existing alignment API; the external decoder receives audio only.
All rows, exclusions, identities and denominators are retained locally.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time

import requests

from Data_Prep.taam_grammar import verse_tokens
from research_os.replay_channel_candidate import local_url
from webapp.channel_calibration import sha256_file


def consonants(word):
    return ''.join(c for c in word if '\u05d0' <= c <= '\u05ea')


def align_words(expected, predicted):
    """Deterministic token edit alignment; insertions/deletions stay visible."""
    a, b = list(map(consonants, expected)), list(map(consonants, predicted))
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): d[i][0] = i
    for j in range(len(b) + 1): d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i-1][j-1] + (a[i-1] != b[j-1]), d[i-1][j]+1, d[i][j-1]+1)
    i, j, pairs = len(a), len(b), []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (a[i-1] != b[j-1]):
            i -= 1; j -= 1; pairs.append((i,j))
        elif i and d[i][j] == d[i-1][j] + 1:
            i -= 1; pairs.append((i,None))
        else:
            j -= 1; pairs.append((None,j))
    return list(reversed(pairs))


def transcription_counts(words, taamim, text):
    tokens = verse_tokens(text)
    counts = Counter(reference_words=len(words), reference_taamim=sum(t is not None for t in taamim))
    for i, j in align_words(words, [w for w, _ in tokens]):
        if i is None:
            counts['insertions'] += 1
        elif j is None:
            counts['deletions'] += 1
        else:
            same_word = consonants(words[i]) == consonants(tokens[j][0])
            counts['matched_words'] += same_word
            counts['substitutions'] += not same_word
            if taamim[i] is not None:
                counts['correct_word_and_taam'] += same_word and taamim[i] == tokens[j][1]
    return counts


def finish_counts(counts):
    c = dict(counts)
    c['word_and_taam_rate'] = counts['correct_word_and_taam'] / counts['reference_taamim'] if counts['reference_taamim'] else None
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--api',type=local_url,required=True)
    ap.add_argument('--external',choices=['none','tiny-smoke','yerushalmi-small'],default='none')
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    for row in rows:
        if not row.get('contiguous_original_phrase') or row.get('spliced'):
            raise ValueError('benchmark requires real unspliced phrases')
        if sha256_file(row['path']) != row['audio_sha256']:
            raise ValueError('manifest audio changed')
    session=requests.Session(); session.trust_env=False
    totals=defaultdict(Counter); per_parasha=defaultdict(lambda:defaultdict(Counter))
    observations=[]
    for k,row in enumerate(rows):
        started=time.perf_counter()
        with Path(row['path']).open('rb') as audio:
            response=session.post(args.api+'/api/check_phrase',data={
                'words':json.dumps(row['words'],ensure_ascii=False), 'taamim':json.dumps(row['taamim']),
                'consent':'false'},files={'audio':('phrase.wav',audio,'audio/wav')},timeout=120)
        response.raise_for_status(); result=response.json()
        counts=Counter(reference_taamim=sum(t is not None for t in row['taamim']),phrases=1)
        counts['alignment_rejected']=not result.get('aligned')
        results=result.get('words',[])
        for i,expected in enumerate(row['taamim']):
            if expected is not None and i<len(results):
                counts['correct_word_and_taam'] += results[i].get('heard')==expected
                counts['taam_ok'] += results[i].get('status')=='ok'
                counts['taam_wrong'] += results[i].get('status')=='wrong'
                counts['taam_unsure'] += results[i].get('status') in ('unknown','unsure','unsupported')
        totals['current_api'].update(counts); per_parasha['current_api'][str(row['parasha'])].update(counts)
        observations.append({'sample_id':row['sample_id'],'parasha':row['parasha'],
            'audio_sha256':row['audio_sha256'],'api_seconds':time.perf_counter()-started,
            'api_result':result,'api_counts':dict(counts)})
        print(json.dumps({'stage':'api','completed':k+1,'total':len(rows)}),flush=True)
    model_identity=None
    if args.external != 'none':
        import torch
        import librosa
        import truststore
        from huggingface_hub import HfApi,snapshot_download
        from transformers import AutoProcessor,AutoModelForSpeechSeq2Seq
        from research_os.try_external_cantillation import VARIANTS,ROOT
        truststore.inject_into_ssl(); torch.set_num_threads(1)
        mid,revision=VARIANTS[args.external]
        info=HfApi().model_info(mid,revision=revision)
        license_id=(info.card_data or {}).get('license')
        if license_id!='apache-2.0': raise ValueError('model card changed; review license')
        print(json.dumps({'stage':'download','model':mid,'revision':revision}),flush=True)
        local=snapshot_download(mid,revision=revision,local_dir=ROOT/'models/external_cantillation'/args.external,
            allow_patterns=['*.json','*.txt','*.safetensors'],max_workers=2)
        processor=AutoProcessor.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        model=AutoModelForSpeechSeq2Seq.from_pretrained(local,local_files_only=True,
            trust_remote_code=False,use_safetensors=True)
        device='mps' if torch.backends.mps.is_available() else 'cpu'; model.eval().to(device)
        model_identity={'model_id':mid,'revision':revision,'weights_sha256':sha256_file(Path(local)/'model.safetensors'),
                        'declared_license':license_id,'device':device}
        for k,(row,obs) in enumerate(zip(rows,observations)):
            waveform,_=librosa.load(row['path'],sr=16000,mono=True)
            if len(waveform)>16000*30: raise ValueError('phrase exceeds decoder limit')
            started=time.perf_counter()
            inputs=processor(waveform,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to(device)
            with torch.inference_mode():
                tokens=model.generate(**inputs,max_new_tokens=200,do_sample=False,language='he',task='transcribe')
            if device=='mps': torch.mps.synchronize()
            text=processor.batch_decode(tokens,skip_special_tokens=True)[0]
            counts=transcription_counts(row['words'],row['taamim'],text)
            totals[args.external].update(counts);per_parasha[args.external][str(row['parasha'])].update(counts)
            obs.update(external_transcription=text,external_counts=dict(counts),external_seconds=time.perf_counter()-started)
            print(json.dumps({'stage':'external','completed':k+1,'total':len(rows)}),flush=True)
    report={'status':'canonical_recognition_diagnostic_not_learner_or_unseen_speaker_test',
        'manifest_sha256':sha256_file(args.manifest),'runner_sha256':sha256_file(__file__),
        'phrases':len(rows),'external_model':model_identity,
        'totals':{name:finish_counts(c) for name,c in totals.items()},
        'by_parasha':{name:{p:finish_counts(c) for p,c in ps.items()} for name,ps in per_parasha.items()},
        'latency_seconds':{key:{'p50':statistics.median(vals),'max':max(vals)}
            for key in ['api_seconds','external_seconds'] if (vals:=[o[key] for o in observations if key in o])},
        'limitations':['Ben13 may overlap external model training','canonical labels not independent human performed labels',
            'API receives expected words for alignment; external decoder receives no expected words or taamim',
            'missed/unmarked reference accents stay in denominator','no learner error precision/recall claim']}
    (args.out/'observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
