"""Read-only diagnostic audit; writes aggregate evidence to an explicit new file.

Does not load checkpoints, train, change production, or export recordings.
Exact feature matching can reconstruct provenance missing from saved NPZ files.
Ambiguous or unmatched rows are reported rather than assigned guessed origins.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def digest(value):
    return hashlib.sha256(value).hexdigest()


def row_key(row):
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=20).hexdigest()


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve earlier evidence: output must be a new path')
    started = time.monotonic()
    meta_path = ROOT / 'features_v3_word/BEN13R_meta.json'
    meta = json.loads(meta_path.read_text())
    features = np.load(ROOT / 'features_v3_word/BEN13R_X.npy', mmap_mode='r')
    assert len(meta) == len(features)
    origins = defaultdict(list)
    for lo in range(0, len(features), 512):
        block = np.asarray(features[lo:lo + 512], dtype=np.float32)
        std = block.std(axis=(1, 2), keepdims=True)
        std[std == 0] = 1.0
        normalized = (block - block.mean(axis=(1, 2), keepdims=True)) / std
        for offset, row in enumerate(normalized):
            origins[row_key(row)].append(lo + offset)
    base = ROOT / 'prepared_data/parasha8_top21_torahhonest'
    names = json.loads((base / 'metadata.json').read_text())['class_names']
    result = {'kind': 'read_only_research_readiness_audit',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runner_sha256': digest(Path(__file__).read_bytes()),
              'source_meta_sha256': digest(meta_path.read_bytes()),
              'prepared_metadata_sha256': digest((base / 'metadata.json').read_bytes()),
              'source_rows': len(meta), 'splits': {}}
    by_split, keys_by_split = {}, {}
    for name in ('train', 'val', 'test'):
        with np.load(base / (name + '.npz')) as data:
            x, y = data['X'], data['y']
        keys, unique_indices = [], []
        unmatched = ambiguous = label_mismatch = 0
        for row, label in zip(x, y):
            key = row_key(row)
            keys.append(key)
            candidates = origins.get(key, [])
            if not candidates:
                unmatched += 1
            elif len(candidates) > 1:
                ambiguous += 1
            else:
                idx = candidates[0]
                unique_indices.append(idx)
                label_mismatch += meta[idx]['taam'] != names[int(label)]
        by_split[name], keys_by_split[name] = unique_indices, set(keys)
        durations = [meta[i]['end'] - meta[i]['start'] for i in unique_indices]
        result['splits'][name] = {
            'rows': len(y), 'exact_unique_origin': len(unique_indices),
            'unmatched': unmatched, 'ambiguous_origin': ambiguous,
            'label_mismatch_unique_origins': int(label_mismatch),
            'unique_feature_rows': len(set(keys)),
            'feature_row_hashes_sha256': digest('\n'.join(keys).encode()),
            'parashot_unique_origins': dict(sorted(Counter(str(meta[i]['parasha']) for i in unique_indices).items())),
            'low_alignment_flag_unique_origins': sum(bool(meta[i].get('low_confidence')) for i in unique_indices),
            'duration_over_2_784s_unique_origins': sum(d > 2.784 for d in durations),
            'duration_percentiles_unique_origins': dict(zip(['p50','p90','p95','p99','max'], np.quantile(durations,[.5,.9,.95,.99,1]).tolist())) if durations else {}}
        del x, y
    result['overlap'] = {}
    for a, b in [('train','val'),('train','test'),('val','test')]:
        groups = lambda s: {(meta[i]['verse_ref'], str(meta[i].get('segment_block'))) for i in by_split[s]}
        ga, gb = groups(a), groups(b)
        result['overlap'][a + '_' + b] = {
            'identical_feature_hashes': len(keys_by_split[a] & keys_by_split[b]),
            'shared_source_blocks_unique_origins': len(ga & gb),
            'a_source_blocks_unique_origins': len(ga), 'b_source_blocks_unique_origins': len(gb)}
    from webapp import server
    first, second = '\u05d1\u05b8\u05bc\u0591\u05e8', '\u05d1\u0591\u05b8\u05bc\u05e8'
    result['unicode_dagesh_probe'] = {
        'canonically_equivalent': unicodedata.normalize('NFD', first) == unicodedata.normalize('NFD', second),
        'first_label': server.expected_channel_label('dagesh', first),
        'second_label': server.expected_channel_label('dagesh', second)}
    actual_changes = 0
    applicable = 0
    for item in meta:
        word = item.get('word_raw', '')
        current = server.expected_channel_label('dagesh', word)
        applicable += current is not None
        actual_changes += current != server.expected_channel_label(
            'dagesh', unicodedata.normalize('NFD', word))
    result['unicode_dagesh_existing_corpus'] = {
        'rows_checked': len(meta), 'applicable_rows': applicable,
        'labels_changed_by_NFD': actual_changes,
        'note': 'Synthetic invariant failure does not establish a cause of current model errors.'}
    module_spec = importlib.util.spec_from_file_location('voice_intake_audit', ROOT / 'Data_Prep/voice_intake.py')
    intake = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(intake)
    probe = {'clip_id':'synthetic-audit-only', 'aligned':True, 'align_score':-1,
             'statuses':['ok'], 'tradition':'ashkenazi'}
    result['intake_function_probe'] = {
        'input_is_synthetic_not_real_recording': True,
        'mismatching_tradition_result': intake.classify_phrase(probe, .99999, {})[0],
        'note':'tests classify_phrase only; does not promote or ingest any sample'}
    result['local_collection'] = {}
    for name, path in [('single_attempts', ROOT/'app_assets/collected_attempts/attempts_log.jsonl'),
                       ('phrases', ROOT/'app_assets/collected_phrases/phrases_log.jsonl')]:
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        result['local_collection'][name] = {'log_exists':path.exists(), 'rows':len(rows),
            'fields':sorted(set().union(*(r.keys() for r in rows))) if rows else []}
    result['seconds'] = round(time.monotonic() - started, 2)
    result['limitations'] = [
        'No audio listening, human correctness labels, or new model inference.',
        'Exact feature matches establish saved feature provenance, not audio uniqueness or split independence from pretraining.',
        'Duration flags do not prove audible truncation or measure its effect on recognition.',
        'Development overlap does not establish train-test leakage.',
        'Single-reader results cannot establish new-reader performance.']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
