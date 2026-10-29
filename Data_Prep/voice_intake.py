"""Produce hash-bound human-only training approvals from local captures.

Model-agreed readings stay quarantined. Actual performed labels, explicit
consent, source identity and unchanged raw audio are all required. This
command never modifies existing raw recordings or capture logs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from .intake_contract import (APPROVAL_KIND, SCHEMA_VERSION, IntakeError,
        annotation_problem, canonical_digest, capture_problem, file_digest,
        load_decisions, load_records, review_entry, utc_now)
except ImportError:
    try:
        from Data_Prep.intake_contract import (APPROVAL_KIND, SCHEMA_VERSION, IntakeError,
            annotation_problem, canonical_digest, capture_problem, file_digest,
            load_decisions, load_records, review_entry, utc_now)
    except ModuleNotFoundError:
        from intake_contract import (APPROVAL_KIND, SCHEMA_VERSION, IntakeError,
            annotation_problem, canonical_digest, capture_problem, file_digest,
            load_decisions, load_records, review_entry, utc_now)

ROOT = Path(__file__).resolve().parent.parent
PHRASES = ROOT/'app_assets/collected_phrases'


def classify_phrase(rec, thresh=None, manual=None):
    """Compatibility helper: model confidence cannot establish human truth."""
    problem = capture_problem(rec)
    if problem:
        return 'rejected', problem
    problem = annotation_problem(rec, (manual or {}).get(rec.get('clip_id')))
    if problem:
        return ('rejected' if problem == 'human_rejected' else 'quarantined'), problem
    return 'verified', 'human_performed_labels_reviewed'


def build_approval(rows, decisions, phrases_dir):
    entries, reasons, seen_audio = [], Counter(), set()
    for rec in rows:
        entry, problem = review_entry(rec, decisions.get(rec['clip_id']), phrases_dir)
        if problem:
            reasons[problem] += 1
            continue
        if entry['audio_sha256'] in seen_audio:
            reasons['duplicate_audio_bytes'] += 1
            continue
        seen_audio.add(entry['audio_sha256'])
        entries.append(entry)
    payload = {'schema_version':SCHEMA_VERSION, 'kind':APPROVAL_KIND,
               'created_at':utc_now(), 'policy':'human-performed-labels-only',
               'entries':entries, 'rejected_reasons':dict(sorted(reasons.items()))}
    payload['manifest_sha256'] = canonical_digest(payload)
    return payload


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phrases-dir', type=Path, default=PHRASES)
    ap.add_argument('--log', type=Path)
    ap.add_argument('--decisions', type=Path)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--promote', action='store_true')
    ap.add_argument('--out', type=Path, help='NEW approval JSON; existing files are preserved')
    ap.add_argument('--review-template', type=Path, help='NEW pending human annotation template, not an approval')
    args = ap.parse_args(argv)
    log = args.log or args.phrases_dir/'phrases_log.jsonl'
    decisions_path = args.decisions or args.phrases_dir/'intake_decisions.json'
    try:
        rows = load_records(log)
        decisions = load_decisions(decisions_path)
        payload = build_approval(rows, decisions, args.phrases_dir)
        if args.review_template:
            clips = {}
            for rec in rows:
                clips[rec['clip_id']] = {
                    'status':'pending', 'label_source':None, 'reviewed_by':None,
                    'reviewed_at':None, 'reading_quality':None,
                    'audio_sha256':rec.get('audio_sha256'), 'record_sha256':canonical_digest(rec),
                    'performed_words':None, 'performed_taamim':None,
                    'note':'Listen independently; do not copy expected text or model statuses as observed labels.'}
            write_new(args.review_template, {'schema_version':SCHEMA_VERSION, 'clips':clips})
        if args.promote:
            payload['capture_log_sha256_at_review'] = file_digest(log) if log.is_file() else None
            payload['decisions_sha256_at_review'] = file_digest(decisions_path) if decisions_path.is_file() else None
            payload['manifest_sha256'] = canonical_digest({k:v for k,v in payload.items() if k != 'manifest_sha256'})
            write_new(args.out or args.phrases_dir/'training_safe.json', payload)
        print(json.dumps({'captured':len(rows), 'human_approved':len(payload['entries']),
            'quarantined_or_rejected':payload['rejected_reasons'], 'promoted':bool(args.promote)},
            ensure_ascii=False, indent=2))
    except (IntakeError, OSError, ValueError, TypeError) as exc:
        ap.exit(2, f'intake refused: {exc}\n')


if __name__ == '__main__':
    main()
