"""Prepare a NEW, auditable BEN13 dataset without recording/duplicate leakage.

All words in a verse and exact normalized-feature duplicates form indivisible
components. Canonical heldout parashot are test only. Components spanning that
boundary, or containing conflicting labels for identical features, are excluded
and recorded. This never overwrites legacy data or changes production weights.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from Data_Prep.holdout import HELD_OUT

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ('train', 'val', 'calibration', 'test')
FEATURE_CONTRACT = {'sample_rate': 16000, 'n_mels': 128, 'fmax': 8000,
                    'max_frames': 87, 'pad_db': -80.0,
                    'normalization': 'per_sample_zero_mean_unit_var',
                    'min_mean_db': -75.0, 'max_pad_ratio': .85,
                    'min_std': .01}


class DataContractError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def normalize_block(block):
    block = np.asarray(block, dtype=np.float32)
    std = block.std(axis=(1, 2), keepdims=True)
    std[std == 0] = 1.0
    return (block - block.mean(axis=(1, 2), keepdims=True)) / std


def feature_hash(row):
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()


def quality_reason(row):
    if row.shape != (128, 87):
        return 'bad_shape'
    if not np.isfinite(row).all():
        return 'contains_nan_or_inf'
    if row.mean() < -75:
        return 'too_silent'
    padding = np.all(row <= -79.9, axis=0) | np.all(row == 0, axis=0)
    if padding.mean() > .85:
        return 'too_much_padding'
    if row.std() < .01:
        return 'no_variance'
    return None


def partition_records(records, class_names, heldout=HELD_OUT, seed=20261028,
                      val_fraction=.1, calibration_fraction=.1):
    """Return assignments and quarantines. Never split connected components.

    Fractions are relative to non-test eligible data. Rare-class components
    are first reserved for training; missing evaluation support is reported
    rather than fabricated by splitting a recording.
    """
    if not 0 < val_fraction < 1 or not 0 < calibration_fraction < 1 or \
            val_fraction + calibration_fraction >= 1:
        raise DataContractError('development/calibration fractions must be positive and sum below one')
    ids = [r['sample_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise DataContractError('sample IDs must be unique')
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owners = {}
    for i, r in enumerate(records):
        if r['label'] not in class_names:
            raise DataContractError('record has an undeclared label')
        for field in ('source_root_id', 'verse_group_id', 'feature_sha256'):
            if not r.get(field):
                raise DataContractError(f'missing grouping field {field}')
            key = (field, r[field])
            if key in owners:
                join(i, owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    labels_by_hash = defaultdict(set)
    for i, r in enumerate(records):
        groups[find(i)].append(i)
        labels_by_hash[r['feature_sha256']].add(r['label'])
    conflicting = {k for k, labels in labels_by_hash.items() if len(labels) > 1}
    heldout = set(map(str, heldout))
    components, quarantine, assigned = [], [], {}
    for indices in groups.values():
        rows = [records[i] for i in indices]
        group_id = 'component:' + stable_hash(sorted(r['sample_id'] for r in rows))
        test_flags = {str(r['parasha']) in heldout for r in rows}
        reason = ('conflicting_duplicate_labels' if any(r['feature_sha256'] in conflicting for r in rows)
                  else 'duplicate_component_crosses_test_boundary' if len(test_flags) > 1 else None)
        component = {'group_id': group_id, 'indices': indices,
                     'counts': Counter(r['label'] for r in rows)}
        for r in rows:
            r['component_id'] = group_id
        if reason:
            quarantine.extend({'sample_id': r['sample_id'], 'reason': reason,
                               'component_id': group_id} for r in rows)
        elif True in test_flags:
            for r in rows:
                assigned[r['sample_id']] = 'test'
        else:
            components.append(component)
    components.sort(key=lambda c: stable_hash([seed, c['group_id']]))
    available = Counter()
    for component in components:
        available.update(component['counts'])
    if any(available[c] == 0 for c in class_names):
        missing = [c for c in class_names if available[c] == 0]
        raise DataContractError(f'no non-test training source for classes: {missing}')

    chosen = {}
    covered = set()
    # Reserve one eligible group for each class, with rare classes first.
    for label in sorted(class_names, key=lambda c: (available[c], c)):
        if label in covered:
            continue
        candidates = [c for c in components if label in c['counts']]
        pick = min(candidates, key=lambda c: (len(c['indices']), c['group_id']))
        chosen[pick['group_id']] = 'train'
        covered.update(pick['counts'])
    fractions = {'train': 1 - val_fraction - calibration_fraction,
                 'val': val_fraction, 'calibration': calibration_fraction}
    counts = {s: Counter() for s in fractions}
    sizes = Counter()
    total = sum(available.values())
    for c in components:
        if c['group_id'] in chosen:
            counts['train'].update(c['counts'])
            sizes['train'] += len(c['indices'])
    for c in components:
        if c['group_id'] in chosen:
            continue
        # Minimize normalized squared deviation from class and total targets.
        def gain(split):
            fraction = fractions[split]
            value = 0.0
            for label, number in c['counts'].items():
                target = available[label] * fraction
                before = counts[split][label] - target
                value += (before ** 2 - (before + number) ** 2) / max(target, 1)
            target = total * fraction
            before = sizes[split] - target
            return value + (before ** 2 - (before + len(c['indices'])) ** 2) / max(target, 1)
        split = max(fractions, key=lambda s: (gain(s), stable_hash([seed, c['group_id'], s])))
        chosen[c['group_id']] = split
        counts[split].update(c['counts'])
        sizes[split] += len(c['indices'])
    for c in components:
        for i in c['indices']:
            assigned[records[i]['sample_id']] = chosen[c['group_id']]
    return assigned, quarantine


def overlap_audit(records, assignments):
    result = {}
    for a, b in itertools.combinations(SPLITS, 2):
        values = {}
        for field in ('sample_id', 'source_root_id', 'verse_group_id',
                      'feature_sha256', 'component_id'):
            left = {r[field] for r in records if assignments.get(r['sample_id']) == a}
            right = {r[field] for r in records if assignments.get(r['sample_id']) == b}
            values[field] = len(left & right)
        result[a + '__' + b] = values
    if any(any(counts.values()) for counts in result.values()):
        raise DataContractError('cross-split overlap remains')
    return result


def verify_dataset(output):
    """Verify the saved contract before downstream training/evaluation.

    The NPZ content hashes cover X without materializing GB of features. Small
    ID/label arrays are checked against each manifest, and group disjointness
    is recomputed from the manifests rather than trusted from a saved report.
    """
    output = Path(output)
    if (output / 'BUILDING').exists() or (output / 'BUILD_FAILED').exists():
        raise DataContractError('dataset build is incomplete')
    metadata = json.loads((output / 'metadata.json').read_text())
    if metadata.get('status') != 'ready':
        raise DataContractError('dataset is not ready')
    for name, expected in metadata['file_sha256'].items():
        if Path(name).name != name or sha256_file(output / name) != expected:
            raise DataContractError(f'dataset artifact changed: {name}')
    records, assignments = [], {}
    for split in SPLITS:
        rows = [json.loads(line) for line in (output / f'{split}_manifest.jsonl').read_text().splitlines()]
        with np.load(output / (split + '.npz')) as data:
            ids, sources, labels = data['sample_ids'], data['source_rows'], data['y']
        if len(rows) != metadata[split + '_samples'] or len(rows) != len(ids):
            raise DataContractError(f'{split}: manifest/NPZ count mismatch')
        for row, sid, source, label in zip(rows, ids, sources, labels):
            if row['sample_id'] != str(sid) or row['source_row'] != int(source) or \
                    row['label'] != metadata['class_names'][int(label)] or row['split'] != split:
                raise DataContractError(f'{split}: row identity mismatch')
            if sid in assignments:
                raise DataContractError('sample ID occurs in multiple splits')
            assignments[str(sid)] = split
            records.append(row)
    overlap_audit(records, assignments)
    if stable_hash(sorted(assignments.items())) != metadata['manifest_assignment_sha256']:
        raise DataContractError('manifest assignment fingerprint changed')
    return metadata


def prepare(features_path, meta_path, classes_path, output, seed=20261028,
            val_fraction=.1, calibration_fraction=.1):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing dataset: {output}')
    features_path, meta_path, classes_path = map(Path, (features_path, meta_path, classes_path))
    meta = json.loads(meta_path.read_text())
    names = json.loads(classes_path.read_text())['class_names']
    if not names or len(names) != len(set(names)):
        raise DataContractError('class list must be nonempty and unique')
    features = np.load(features_path, mmap_mode='r')
    if features.shape != (len(meta), 128, 87):
        raise DataContractError('BEN13 metadata/features shape mismatch')
    metadata_hash = sha256_file(meta_path)
    records, exclusions = [], []
    for lo in range(0, len(meta), 512):
        block = np.asarray(features[lo:lo + 512], dtype=np.float32)
        valid = []
        for j, row in enumerate(block):
            index = lo + j
            m = meta[index]
            sample_id = f'BEN13R:{metadata_hash[:16]}:{index:06d}'
            reason = 'other_class' if m.get('taam') not in names else quality_reason(row)
            if not m.get('verse_ref') or not m.get('segment_block') or not m.get('parasha'):
                reason = 'missing_source_metadata'
            if reason:
                exclusions.append({'sample_id': sample_id, 'source_row': index, 'reason': reason})
            else:
                valid.append((j, sample_id))
        if not valid:
            continue
        normalized = normalize_block(block[[j for j, _ in valid]])
        for (j, sample_id), row in zip(valid, normalized):
            index = lo + j
            m = meta[index]
            records.append({'sample_id': sample_id, 'source_row': index,
                'speaker_id': 'BEN13R', 'session_id': None,
                'source_root_id': f"BEN13R:{m['verse_ref']}:{m['segment_block']}",
                'verse_group_id': f"BEN13R:{m['verse_ref']}",
                'feature_sha256': feature_hash(row), 'parasha': str(m['parasha']),
                'verse_ref': m['verse_ref'], 'segment_block': str(m['segment_block']),
                'word_raw': m.get('word_raw'), 'word_clean': m.get('word_clean'),
                'start': m.get('start'), 'end': m.get('end'),
                'label': m['taam'], 'label_origin': 'expected_from_text',
                'tradition': 'sephardic_jerusalem', 'low_confidence': bool(m.get('low_confidence')),
                'alignment_method': m.get('align'), 'alignment_score': m.get('ctc_score')})
    assignments, quarantine = partition_records(records, names, seed=seed,
        val_fraction=val_fraction, calibration_fraction=calibration_fraction)
    overlaps = overlap_audit(records, assignments)
    split_rows = {s: [r for r in records if assignments.get(r['sample_id']) == s] for s in SPLITS}
    if any(not rows for rows in split_rows.values()):
        raise DataContractError('insufficient independent groups to populate all four splits')
    # Compute all contract decisions before allocating a new immutable directory.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (output / 'BUILDING').write_text('Do not train until metadata.json status is ready.\n')
    try:
        provenance = {'features_sha256': sha256_file(features_path),
                      'source_metadata_sha256': metadata_hash,
                      'class_metadata_sha256': sha256_file(classes_path),
                      'builder_sha256': sha256_file(__file__),
                      'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
        file_hashes, distribution = {}, {}
        label_to_idx = {name: i for i, name in enumerate(names)}
        for split, rows in split_rows.items():
            indices = np.array([r['source_row'] for r in rows], dtype=np.int64)
            x = normalize_block(features[indices])[:, None]
            y = np.array([label_to_idx[r['label']] for r in rows], dtype=np.int64)
            ids = np.array([r['sample_id'] for r in rows])
            with (output / (split + '.npz')).open('xb') as stream:
                np.savez_compressed(stream, X=x, y=y, sample_ids=ids, source_rows=indices)
            del x, y
            manifest = output / (split + '_manifest.jsonl')
            with manifest.open('x', encoding='utf-8') as stream:
                for r in rows:
                    stream.write(json.dumps({**r, 'split': split}, ensure_ascii=False, sort_keys=True) + '\n')
            file_hashes[split + '.npz'] = sha256_file(output / (split + '.npz'))
            file_hashes[manifest.name] = sha256_file(manifest)
            distribution[split] = dict(Counter(r['label'] for r in rows))
        (output / 'excluded.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in exclusions + quarantine))
        file_hashes['excluded.jsonl'] = sha256_file(output / 'excluded.jsonl')
        metadata = {'schema_version': 1, 'status': 'ready', 'scenario': output.name,
                    'split': 'source_connected_components', 'readers': ['BEN13R'],
                    'class_names': names, 'label_to_idx': label_to_idx,
                    'num_classes': len(names), 'input_shape': [1,128,87],
                    'normalization': FEATURE_CONTRACT['normalization'],
                    'feature_contract': FEATURE_CONTRACT, 'seed': seed,
                    'heldout_parashot': list(HELD_OUT),
                    'fractions_of_non_test': {'val': val_fraction, 'calibration': calibration_fraction},
                    'source_rows': len(meta), 'eligible_before_quarantine': len(records),
                    'exclusion_counts': dict(Counter(r['reason'] for r in exclusions + quarantine)),
                    'class_distribution': distribution,
                    'missing_classes_by_split': {s: [c for c in names if distribution[s].get(c, 0) == 0] for s in SPLITS},
                    'overlap_audit': overlaps, 'provenance': provenance,
                    'file_sha256': file_hashes,
                    'manifest_assignment_sha256': stable_hash(sorted(assignments.items())),
                    'limitations': ['Single BEN13 voice; no claim of new-reader generalization.',
                        'Labels originate in canonical text and are not human-audited performance labels.',
                        'Historical canonical test has been inspected previously; not a fresh sealed test.',
                        'Exact feature duplicates only; acoustic near-duplicate detection is not covered.',
                        'Unknown original recording sessions cannot be separated beyond available verse/source metadata.'],
                    'build_seconds': round(time.monotonic() - started, 3)}
        for split, rows in split_rows.items():
            metadata[split + '_samples'] = len(rows)
        metadata['dataset_fingerprint'] = stable_hash({k: metadata[k] for k in (
            'feature_contract','seed','heldout_parashot','fractions_of_non_test','provenance',
            'manifest_assignment_sha256')})
        (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
        (output / 'BUILDING').unlink()
        return metadata
    except BaseException:
        (output / 'BUILD_FAILED').write_text('Incomplete build; preserve for diagnosis, choose a new output directory.\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=ROOT/'features_v3_word/BEN13R_X.npy')
    parser.add_argument('--meta', type=Path, default=ROOT/'features_v3_word/BEN13R_meta.json')
    parser.add_argument('--classes-from', type=Path, default=ROOT/'prepared_data/parasha8_top21_torahhonest/metadata.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--calibration-fraction', type=float, default=.1)
    args = parser.parse_args()
    report = prepare(args.features, args.meta, args.classes_from, args.out,
                     args.seed, args.val_fraction, args.calibration_fraction)
    print(json.dumps({k: report[k] for k in ('status', 'scenario', 'dataset_fingerprint',
        'train_samples','val_samples','calibration_samples','test_samples',
        'exclusion_counts','missing_classes_by_split','overlap_audit','build_seconds')}, indent=2))


if __name__ == '__main__':
    main()
