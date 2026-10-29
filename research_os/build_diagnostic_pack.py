"""Build a local-only, blinded human-review pack and real phrase benchmark.

No model inference or human labels are invented. All source joins are exact,
and repeated paths are checked for audio/text equivalence. Outputs must live
inside the ignored results directory and a new output directory is required.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import unicodedata

import numpy as np
import soundfile as sf

from Data_Prep.holdout import HELD_OUT
from research_os.source_audio import SourceConflict, SourceResolver, sha256_file

ROOT = Path(__file__).resolve().parent.parent
STRATA = [('weak_taam', 60), ('rare_taam', 40), ('boundary_or_alignment', 40),
          ('grammar_probe', 40), ('random_control', 20)]
WEAK = {'Segol', 'Etnachta', 'Qadma', 'Munach'}
RARE = {'Pazer', 'Telisha_Gedola', 'Telisha_Qetana', 'Zarqa', 'Yetiv', 'Gershayim', 'Segol'}


def stable_id(prefix, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return prefix + hashlib.sha256(payload).hexdigest()[:20]


def json_lines(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def load_candidates(root):
    meta = json.loads((root / 'features_v3_word/BEN13R_meta.json').read_text())
    manifest = json.loads((root / 'data/ben13_reading/manifest_yerushalmi.json').read_text())
    candidates, blocks = [], defaultdict(list)
    for index, row in enumerate(meta):
        if str(row.get('parasha')) not in set(HELD_OUT):
            continue
        try:
            b, c, v = row['verse_ref'].split('.')
            start, end = float(row['start']), float(row['end'])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                continue
            key = (b, c, v, str(row['segment_block']))
            value = {'meta_index': index, 'meta': row, 'key': key}
            candidates.append(value)
            blocks[key].append(value)
        except (KeyError, ValueError, TypeError):
            continue
    for rows in blocks.values():
        rows.sort(key=lambda r: (r['meta']['start'], r['meta_index']))
    return meta, manifest, candidates, blocks


def belongs(item, stratum):
    row = item['meta']
    if stratum == 'weak_taam':
        return row.get('taam') in WEAK
    if stratum == 'rare_taam':
        return row.get('taam') in RARE
    if stratum == 'boundary_or_alignment':
        return bool(row.get('low_confidence')) or row['end'] - row['start'] > 2.784
    if stratum == 'grammar_probe':
        word = row.get('word_raw', '')
        return any(c in word for c in ('\u05b0', '\u05bc', '\u05e2', '\u05d0'))
    return True


def stratified_order(items, rng):
    buckets = defaultdict(list)
    for item in items:
        buckets[str(item['meta']['parasha'])].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets, key=int)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                yield buckets[key].pop()


def build_pack(root, output, seed=20261028, max_words=200, per_parasha=5, benchmark_blocks=None):
    # benchmark_blocks: optional 'book.chapter.verse:block' allowlist, e.g. the
    # blocks an external model never trained on. None keeps every block eligible.
    allow = None if benchmark_blocks is None else frozenset(benchmark_blocks)
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / 'results'):
        raise ValueError('output must stay inside the local ignored results directory')
    if not 1 <= max_words <= 200 or not 1 <= per_parasha <= 20:
        raise ValueError('use 1..200 review words and 1..20 benchmark phrases per parasha')
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing review packet')
    meta, manifest, candidates, blocks = load_candidates(root)
    resolver = SourceResolver(root, manifest)
    rng = random.Random(seed)
    output.mkdir(parents=True)
    (output / 'audio').mkdir()
    source_cache, rejected = {}, Counter()

    def source_for(item):
        source = resolver.resolve(item['key'])
        expected = tuple((unicodedata.normalize('NFD', r['meta']['word_raw']), r['meta']['taam'])
                         for r in blocks[item['key']])
        if expected != source['tokenization']:
            raise SourceConflict('metadata_source_tokenization_mismatch')
        sha = source['source_audio_sha256']
        if sha not in source_cache:
            waveform, sr = sf.read(source['path'], dtype='float32', always_2d=True)
            waveform = waveform.mean(axis=1)
            if not len(waveform) or not np.isfinite(waveform).all():
                raise SourceConflict('source_audio_invalid')
            seconds = len(waveform) / sr
            if not 0.2 <= seconds <= 30:
                raise SourceConflict('source_audio_duration_outside_0_2_to_30s')
            relative = 'audio/source-' + sha[:20] + '.wav'
            sf.write(output / relative, waveform, sr, subtype='PCM_16')
            source_cache[sha] = (waveform, sr, relative, sha256_file(output / relative))
        waveform, sr, relative, wav_sha = source_cache[sha]
        return source, waveform, sr, relative, wav_sha

    reviews, used, accounting = [], set(), {}
    for stratum, default_target in STRATA:
        target = min(default_target, max_words - len(reviews))
        selected = 0
        pool = [item for item in candidates if belongs(item, stratum)]
        for item in stratified_order(pool, rng):
            if selected >= target:
                break
            row = item['meta']
            try:
                source, waveform, sr, context_relative, _ = source_for(item)
                sample_id = stable_id('word-', [source['source_audio_sha256'],
                    row['start'], row['end'], row['word_raw'], row.get('block_pos')])
                if sample_id in used:
                    continue
                low = max(0.0, float(row['start']) - 0.15)
                high = min(len(waveform) / sr, float(row['end']) + 0.20)
                audio = waveform[int(low * sr):int(high * sr)]
                if len(audio) < sr * .1:
                    raise SourceConflict('word_cut_empty_or_short')
                relative = f'audio/{sample_id}.wav'
                sf.write(output / relative, audio, sr, subtype='PCM_16')
                reviews.append({'schema_version': 1, 'sample_id': sample_id,
                    'source_root_id': source['source_root_id'], 'speaker_id': 'BEN13R',
                    'tradition': 'sephardic-jerusalem', 'parasha': str(row['parasha']),
                    'verse': row['verse_ref'], 'block': str(row['segment_block']),
                    'source_aliases': source['aliases'], 'source_audio_sha256': source['source_audio_sha256'],
                    'audio_sha256': sha256_file(output / relative), 'path': str(output / relative),
                    'audio_relative_path': relative, 'context_relative_path': context_relative,
                    'source_start': low, 'source_end': high, 'aligned_start': row['start'],
                    'aligned_end': row['end'], 'source_duration': len(waveform) / sr,
                    'sample_rate': sr, 'selection_stratum': stratum,
                    'selection_is_diagnostic_not_population_sample': True,
                    'expected_word': row['word_raw'], 'expected_taam': row['taam'],
                    'label_provenance': 'expected_from_source_text_not_human_performed',
                    'performed_taam': None, 'performed_transcript': None, 'human_correct': None,
                    'annotation_status': 'pending', 'source_meta_index': item['meta_index'],
                    'evaluation_scope': 'historical_single_reader_diagnostic_not_sealed_test',
                    'low_alignment_flag': bool(row.get('low_confidence')),
                    'bounds_exceed_audio': row['end'] > len(waveform) / sr + .02,
                    'source_rights_status': 'not_verified_local_diagnostic_only',
                    'possible_external_model_training_overlap': True})
                used.add(sample_id)
                selected += 1
            except (SourceConflict, RuntimeError, sf.LibsndfileError, ValueError) as exc:
                rejected[str(exc)] += 1
        accounting[stratum] = {'requested': target, 'selected': selected, 'candidate_rows': len(pool)}
        if len(reviews) >= max_words:
            break

    benchmarks, roots = [], set()
    for parasha in sorted(HELD_OUT, key=int):
        pool = [rows[0] for rows in blocks.values()
                if len(rows) >= 2 and str(rows[0]['meta']['parasha']) == parasha
                and all(r['meta'].get('read_in_sequence') for r in rows)
                and (allow is None or '.'.join(rows[0]['key'][:3]) + ':' + rows[0]['key'][3] in allow)]
        rng.shuffle(pool)
        selected = 0
        for item in pool:
            if selected >= per_parasha:
                break
            try:
                source, waveform, sr, relative, wav_sha = source_for(item)
                if source['source_root_id'] in roots:
                    continue
                rows = blocks[item['key']]
                sample_id = stable_id('phrase-', [source['source_audio_sha256'], source['tokenization']])
                benchmarks.append({'schema_version': 1, 'sample_id': sample_id,
                    'source_root_id': source['source_root_id'], 'speaker_id': 'BEN13R',
                    'tradition': 'sephardic-jerusalem', 'parasha': parasha,
                    'verse': rows[0]['meta']['verse_ref'], 'block': item['key'][3],
                    'path': str(output / relative), 'audio_relative_path': relative,
                    'source_audio_sha256': source['source_audio_sha256'], 'audio_sha256': wav_sha,
                    'sample_rate': sr, 'duration_seconds': len(waveform) / sr,
                    'words': [r['meta']['word_raw'] for r in rows],
                    'taamim': [r['meta']['taam'] for r in rows],
                    'contiguous_original_phrase': True, 'spliced': False,
                    'expected_labels_are_not_human_verified_performed': True,
                    'evaluation_scope': 'historical_single_reader_diagnostic_not_sealed_test',
                    'source_rights_status': 'not_verified_local_diagnostic_only',
                    'possible_external_model_training_overlap': True})
                roots.add(source['source_root_id'])
                selected += 1
            except (SourceConflict, RuntimeError, sf.LibsndfileError, ValueError) as exc:
                rejected[str(exc)] += 1
        accounting[f'benchmark_parasha_{parasha}'] = {'requested': per_parasha, 'selected': selected}
    json_lines(output / 'review_manifest.jsonl', reviews)
    json_lines(output / 'benchmark_manifest.jsonl', benchmarks)
    annotations = [{'schema_version': 1, 'sample_id': row['sample_id'],
                    'audio_sha256': row['audio_sha256'], 'reviewer_id': None,
                    'blind_performed_taam': None, 'blind_performed_transcript': None,
                    'human_correct': None, 'error_type': None, 'confidence': None,
                    'notes': '', 'annotation_status': 'pending'} for row in reviews]
    json_lines(output / 'annotations.template.jsonl', annotations)
    report = {'schema_version': 1, 'status': 'ready_for_local_human_review_not_gold',
              'seed': seed, 'review_units': len(reviews), 'benchmark_phrases': len(benchmarks),
              'review_parashot': dict(Counter(r['parasha'] for r in reviews)),
              'benchmark_parashot': dict(Counter(r['parasha'] for r in benchmarks)),
              'benchmark_block_allowlist': None if allow is None else {
                  'blocks': len(allow),
                  'sha256': hashlib.sha256('\n'.join(sorted(allow)).encode()).hexdigest()},
              'selection_accounting': accounting, 'rejections': dict(rejected),
              'rejection_counts_are_candidate_attempts_not_distinct_sources': True,
              'source_resolution': resolver.report(),
              'review_manifest_sha256': sha256_file(output / 'review_manifest.jsonl'),
              'benchmark_manifest_sha256': sha256_file(output / 'benchmark_manifest.jsonl'),
              'source_manifest_sha256': sha256_file(root / 'data/ben13_reading/manifest_yerushalmi.json'),
              'feature_metadata_sha256': sha256_file(root / 'features_v3_word/BEN13R_meta.json'),
              'limitations': ['single Ben13 voice', 'diagnostic strata are intentionally biased',
                              'expected labels are not observed performed labels',
                              'no models run and no known failures or successes inferred',
                              'source rights not verified; local use only, do not publish pack',
                              'possible overlap with external model training audio']}
    (output / 'selection_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'review.html').write_text(review_html(reviews, report), encoding='utf-8')
    return report


def review_html(rows, report):
    # JSON strings are escaped against script termination; no remote resource.
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    pack_id = report['review_manifest_sha256']
    return '''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Taamify — ביקורת שמע</title>
<style>body{font:18px system-ui;max-width:850px;margin:30px auto;padding:20px;background:#f6f7fa;color:#182433}section{background:white;padding:24px;border-radius:16px}button,input,select,textarea{font:inherit;padding:9px;margin:6px;max-width:95%}label{display:block;margin:12px 0}textarea{width:90%}audio{display:block;width:100%;margin:15px 0}.sub{color:#56657b;font-size:15px}#expected{background:#eff5ff;padding:16px;margin:14px 0}nav{display:flex;flex-wrap:wrap;gap:8px}button{cursor:pointer}#status{min-height:24px}</style>
<h1>ביקורת ביצוע — האזנה לפני חשיפת הציפייה</h1>
<p class="sub">חומר מקומי לפיתוח. זכויות המקור טרם אומתו להפצה. הקובץ פועל ללא שליחת נתונים לרשת; שמירת התיוג בדפדפן ובקובץ שיורד למחשב.</p>
<label>מזהה בודק בדוי <input id="reviewer" placeholder="למשל reviewer-01"></label>
<nav><button id="previous">הקודם</button><span id="counter"></span><button id="next">הבא</button><button id="export">הורדת התיוגים שבוצעו</button></nav>
<section><h2 id="sample"></h2><p>ראשית האזן למילה ולהקשר וסמן מה נשמע. תחזיות מודל אינן מוצגות.</p>
<p>המילה בחיתוך אבחון</p><audio id="wordAudio" controls preload="none"></audio>
<p>המשפט המקורי המלא</p><audio id="contextAudio" controls preload="none"></audio>
<label>איזה טעם או מימוש נשמע? <input id="performed" placeholder="או: לא ניתן להכריע"></label>
<label>תמלול/תיאור הביצוע <textarea id="transcript"></textarea></label>
<button id="reveal">נעילת ההאזנה הראשונה וחשיפת הטקסט המצופה</button>
<div id="expected" hidden></div>
<div id="assessment" hidden><label>האם הביצוע תקין? <select id="correct"><option value="">בחר</option><option value="true">תקין</option><option value="false">שגוי</option><option value="unclear">לא ניתן להכריע</option></select></label>
<label>סוג הטעות, אם יש <input id="errorType" placeholder="טעם / הטעמה / הגייה / חיתוך / אחר"></label>
<label>ביטחון בהערכה <select id="confidence"><option value="">בחר</option><option>גבוה</option><option>בינוני</option><option>נמוך</option></select></label>
<label>הערות <textarea id="notes"></textarea></label><button id="save">שמירת תיוג</button></div>
<p id="status" role="status"></p></section>
<script>const rows=PAYLOAD, packId=PACK_ID, key="taamify-review-"+packId;
const $=x=>document.getElementById(x);let i=0,store={};try{store=JSON.parse(localStorage.getItem(key)||"{}")}catch(e){}
const persist=()=>{try{localStorage.setItem(key,JSON.stringify(store))}catch(e){$("status").textContent="השמירה בדפדפן נכשלה; הורד את קובץ התיוגים"}};
function render(){const r=rows[i],a=store[r.sample_id]||{};$("counter").textContent=(i+1)+" / "+rows.length;$("sample").textContent="דוגמה "+(i+1);$("wordAudio").src=r.audio_relative_path;$("contextAudio").src=r.context_relative_path;$("performed").value=a.blind_performed_taam||"";$("transcript").value=a.blind_performed_transcript||"";$("performed").disabled=!!a.blind_locked_at;$("transcript").disabled=!!a.blind_locked_at;$("expected").hidden=!a.blind_locked_at;$("assessment").hidden=!a.blind_locked_at;$("reveal").disabled=!!a.blind_locked_at;$("expected").textContent="מצופה: "+r.expected_word+" · "+r.expected_taam;$("correct").value=a.human_correct===true?"true":a.human_correct===false?"false":a.human_correct===null&&a.annotation_status==="human_reviewed"?"unclear":"";$("errorType").value=a.error_type||"";$("confidence").value=a.confidence||"";$("notes").value=a.notes||"";$("status").textContent=a.annotation_status==="human_reviewed"?"התיוג נשמר":"";if(a.reviewer_id&&!$("reviewer").value)$("reviewer").value=a.reviewer_id;}
$("reveal").onclick=()=>{if(!$("reviewer").value.trim()||!$("performed").value.trim()){$("status").textContent="מלא מזהה בודק ותיאור ביצוע לפני חשיפת הציפייה";return}const r=rows[i];store[r.sample_id]={schema_version:1,sample_id:r.sample_id,audio_sha256:r.audio_sha256,reviewer_id:$("reviewer").value.trim(),blind_performed_taam:$("performed").value.trim(),blind_performed_transcript:$("transcript").value.trim(),blind_locked_at:new Date().toISOString(),annotation_status:"blind_phase_locked"};persist();render()};
$("save").onclick=()=>{const r=rows[i],a=store[r.sample_id];if(!a?.blind_locked_at||!$("correct").value||!$("confidence").value){$("status").textContent="השלם תקינות וביטחון";return}Object.assign(a,{human_correct:$("correct").value==="unclear"?null:$("correct").value==="true",error_type:$("errorType").value.trim()||null,confidence:$("confidence").value,notes:$("notes").value.trim(),annotation_status:"human_reviewed",reviewed_at:new Date().toISOString(),pack_sha256:packId});persist();render()};
$("previous").onclick=()=>{i=Math.max(0,i-1);render()};$("next").onclick=()=>{i=Math.min(rows.length-1,i+1);render()};
$("export").onclick=()=>{const done=Object.values(store).filter(a=>a.annotation_status==="human_reviewed"),blob=new Blob([done.map(a=>JSON.stringify(a)).join("\\n")+(done.length?"\\n":"")],{type:"application/x-ndjson"}),link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="taamify-human-annotations.jsonl";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);$("status").textContent="יוצאו "+done.length+" תיוגים; אין אישור אוטומטי לשימוש באימון"};render();</script></html>'''.replace('PAYLOAD', payload).replace('PACK_ID', json.dumps(pack_id))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--max-words', type=int, default=200)
    ap.add_argument('--per-parasha', type=int, default=5)
    ap.add_argument('--benchmark-blocks', type=Path,
                    help="JSON list of 'book.chapter.verse:block' eligible for benchmark phrases")
    args = ap.parse_args()
    blocks = json.loads(args.benchmark_blocks.read_text()) if args.benchmark_blocks else None
    start = time.monotonic()
    result = build_pack(ROOT, args.out, args.seed, args.max_words, args.per_parasha, blocks)
    print(json.dumps({**result, 'seconds': round(time.monotonic() - start, 2)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
