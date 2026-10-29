"""Exact join of Ben13 Jerusalem blocks to the Technion cantillation split.

The Technion Jerusalem Whisper model (cantillation/Teamim-small_..._nusach-
yerushalmi_date-24-07-2024) was trained on Ben13 — the same recording as ours.
Its split files are committed in staviv/cantillation@6f16c5f (2024-07-17,
jsons/{train,validation,test}_data.json); no commit touched jsons/ before the
model's date. Rows are keyed by the Ben13 file name, so the join is exact:
parasha, book+chapter+verse, block. No duration or text matching.

Reads a local clone only. Writes a report and an allowlist of blocks from our
held-out parashot that the Technion model never trained on, for
build_diagnostic_pack --benchmark-blocks.

    git clone https://github.com/staviv/cantillation /tmp/x/cantillation
    git -C /tmp/x/cantillation show 6f16c5f:jsons/train_data.json > split/train_data.json  # etc.
    venv/bin/python -m research_os.technion_split_overlap --split-dir split \
        --out results/research_os/benchmarks/technion_overlap_NEW.json \
        --allowlist results/research_os/benchmarks/technion_heldout_blocks_NEW.json
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from Data_Prep.holdout import HELD_OUT

ROOT = Path(__file__).resolve().parent.parent
TECHNION_COMMIT = '6f16c5f'
PATTERN = re.compile(r'yerushalmi_(\d+)-[^_]+_(\d+)_(\d)(\d\d)(\d\d)_(\d+)\.mp3$')


def letters(text):
    return [w for w in ''.join(c for c in unicodedata.normalize('NFD', text)
                               if c.isalpha() or c.isspace()).split()]


def load_split(split_dir):
    membership, text = defaultdict(set), {}
    for split in ('train', 'validation', 'test'):
        data = json.loads((split_dir / f'{split}_data.json').read_text())
        for path, words in zip(data['yerushalmi'], data['text']):
            m = PATTERN.search(path)
            if not m:
                continue
            p, _, b, c, v, block = m.groups()
            key = (int(p), f'{int(b)}.{int(c)}.{int(v)}', int(block))
            membership[key].add(split)
            text[key] = words
    return membership, text


def classify(splits):
    if 'train' in splits:
        return 'technion_train'
    return 'technion_' + '+'.join(sorted(splits)) if splits else 'absent_from_technion_split'


def benchmark_by_split(membership, text, manifest, observations):
    obs = {o['sample_id']: o for o in json.loads(observations.read_text())}
    groups, checked, mismatched = defaultdict(Counter), 0, []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row['parasha']), row['verse'], int(row['block']))
        if key in text:
            checked += 1
            if not all(w in letters(text[key]) for w in letters(' '.join(row['words']))):
                mismatched.append(row['sample_id'])
        o, g = obs[row['sample_id']], groups[classify(membership.get(key, set()))]
        g['phrases'] += 1
        g['reference_taamim'] += o['external_counts']['reference_taamim']
        g['external_correct'] += o['external_counts']['correct_word_and_taam']
        # An alignment-rejected phrase has no per-word counts: it scores zero.
        g['api_correct'] += o['api_counts'].get('correct_word_and_taam', 0)
    out = {k: dict(v, external_rate=round(v['external_correct'] / v['reference_taamim'], 4),
                   api_rate=round(v['api_correct'] / v['reference_taamim'], 4))
           for k, v in groups.items()}
    return out, {'checked': checked, 'text_mismatch': mismatched}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allowlist', type=Path, required=True)
    ap.add_argument('--benchmark-manifest', type=Path)
    ap.add_argument('--benchmark-observations', type=Path)
    args = ap.parse_args()
    for path in (args.out, args.allowlist):
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')

    membership, text = load_split(args.split_dir)
    ben13 = json.loads((ROOT / 'data/ben13_reading/manifest_yerushalmi.json').read_text())
    held_out = {int(p) for p in HELD_OUT}
    per_parasha, allow = defaultdict(Counter), set()
    for e in ben13:
        p = int(e['parasha'])
        if p not in held_out:
            continue
        verse = f"{int(e['book'])}.{int(e['perek'])}.{int(e['pasuk'])}"
        label = classify(membership.get((p, verse, int(e['block'])), set()))
        per_parasha[p][label] += 1
        if label in ('technion_test', 'technion_validation'):
            allow.add(f"{verse}:{int(e['block'])}")

    report = {
        'technion_source': f'staviv/cantillation@{TECHNION_COMMIT} jsons/*_data.json',
        'split_file_sha256': {s: hashlib.sha256((args.split_dir / f'{s}_data.json').read_bytes()).hexdigest()
                              for s in ('train', 'validation', 'test')},
        'technion_jerusalem_rows': len(membership),
        'technion_rows_in_more_than_one_split': sum(len(v) > 1 for v in membership.values()),
        'held_out_parashot_blocks': {str(p): dict(c) for p, c in sorted(per_parasha.items())},
        'held_out_parashot_total': dict(sum(per_parasha.values(), Counter())),
        'allowlist_blocks': len(allow),
        'allowlist_note': 'technion_validation selected their checkpoint; technion_test was never used',
    }
    if args.benchmark_manifest:
        report['benchmark_by_technion_split'], report['benchmark_text_check'] = benchmark_by_split(
            membership, text, args.benchmark_manifest, args.benchmark_observations)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    args.allowlist.write_text(json.dumps(sorted(allow), indent=0) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
