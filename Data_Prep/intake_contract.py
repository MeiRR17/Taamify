"""Consent and human-performed-label provenance for local training intake."""
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

SCHEMA_VERSION = 2
CONSENT_VERSION = 'taamify-local-research-v1'
TRADITION = 'sephardic-jerusalem'
APPROVAL_KIND = 'taamify.training-approval'
PSEUDONYM = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{2,95}\Z')
CLIP_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
TAAMIM = frozenset(('Tipecha', 'Mercha', 'Munach', 'Zaqef_Qatan', 'Siluk',
    'Etnachta', 'Pashta', 'Mahapakh', 'Tevir', 'Revia', 'Qadma', 'Geresh',
    'Darga', 'Zaqef_Gadol', 'Gershayim', 'Telisha_Qetana', 'Zarqa', 'Segol',
    'Yetiv', 'Telisha_Gedola', 'Pazer'))


class IntakeError(ValueError):
    """Invalid input: callers must not write training output."""


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_records(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows, seen = [], set()
    try:
        for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise IntakeError(f'log line {n}: expected an object')
            cid = item.get('clip_id')
            if not isinstance(cid, str) or not CLIP_ID.fullmatch(cid):
                raise IntakeError(f'log line {n}: invalid clip_id')
            if cid in seen:
                raise IntakeError(f'log line {n}: duplicate clip_id')
            seen.add(cid)
            rows.append(item)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(f'cannot read capture log: {exc}') from exc
    return rows


def load_decisions(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(f'cannot read human annotations: {exc}') from exc
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA_VERSION:
        raise IntakeError('annotations require schema_version=2; legacy tier strings are not approvals')
    if not isinstance(value.get('clips'), dict):
        raise IntakeError('annotations must contain a clips object')
    return value['clips']


def capture_problem(rec):
    if not isinstance(rec.get('clip_id'), str) or not CLIP_ID.fullmatch(rec['clip_id']):
        return 'clip_id_invalid'
    if rec.get('schema_version') != SCHEMA_VERSION:
        return 'capture_schema_unsupported'
    if rec.get('consent') is not True or rec.get('consent_version') != CONSENT_VERSION:
        return 'explicit_consent_missing_or_unsupported'
    scope = rec.get('consent_scope')
    if not isinstance(scope, list) or not {'local_research', 'training'} <= {x for x in scope if isinstance(x, str)}:
        return 'training_consent_missing'
    for field in ('voice_id', 'session_id'):
        value = rec.get(field)
        if not isinstance(value, str) or not PSEUDONYM.fullmatch(value) or value.lower() in {'anon', 'anonymous', 'unknown'}:
            return f'{field}_missing_or_invalid'
    if rec.get('tradition') != TRADITION:
        return 'tradition_missing_or_ineligible'
    if not isinstance(rec.get('source_root_id'), str) or not rec['source_root_id'].strip():
        return 'source_root_id_missing'
    if not isinstance(rec.get('audio_sha256'), str) or not SHA256.fullmatch(rec['audio_sha256']):
        return 'capture_audio_hash_missing'
    words, taamim = rec.get('words'), rec.get('taamim')
    if not isinstance(words, list) or not words or not all(isinstance(x, str) and x.strip() for x in words):
        return 'expected_words_invalid'
    if not isinstance(taamim, list) or len(words) != len(taamim) or any(
            t is not None and not isinstance(t, str) for t in taamim):
        return 'expected_taamim_invalid'
    return None


def annotation_problem(rec, annotation):
    if not isinstance(annotation, dict):
        return 'human_annotation_missing'
    if annotation.get('status') == 'rejected':
        return 'human_rejected'
    if annotation.get('status') != 'approved':
        return 'human_review_pending'
    if annotation.get('label_source') not in {'human-reviewed', 'adjudicated'}:
        return 'performed_labels_not_human_reviewed'
    reviewer = annotation.get('reviewed_by')
    if not isinstance(reviewer, str) or not PSEUDONYM.fullmatch(reviewer) or reviewer.lower() in {'anon', 'anonymous', 'unknown'}:
        return 'reviewer_missing'
    try:
        stamp = datetime.fromisoformat(annotation['reviewed_at'].replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            return 'review_timestamp_missing_timezone'
    except (ValueError, TypeError, KeyError, AttributeError):
        return 'review_timestamp_invalid'
    if annotation.get('audio_sha256') != rec.get('audio_sha256'):
        return 'annotation_audio_hash_mismatch'
    if annotation.get('record_sha256') != canonical_digest(rec):
        return 'annotation_record_hash_mismatch'
    words, taamim = annotation.get('performed_words'), annotation.get('performed_taamim')
    if not isinstance(words, list) or not words or not all(isinstance(x, str) and x.strip() for x in words):
        return 'performed_words_missing'
    if not isinstance(taamim, list) or len(taamim) != len(words):
        return 'performed_taamim_misaligned'
    if any(t is not None and (not isinstance(t, str) or t not in TAAMIM) for t in taamim) or not any(isinstance(t, str) and t in TAAMIM for t in taamim):
        return 'performed_taamim_unsupported'
    # Substitution/omission errors need separately reviewed alignment. Keep
    # them for the error dataset instead of forcing expected words over them.
    if words != rec.get('words'):
        return 'performed_words_differ_requires_error_dataset'
    if annotation.get('reading_quality') != 'clear':
        return 'performed_reading_not_clear'
    return None


def review_entry(rec, annotation, phrases_dir):
    problem = capture_problem(rec) or annotation_problem(rec, annotation)
    if problem:
        return None, problem
    audio = Path(phrases_dir) / f"{rec['clip_id']}.wav"
    if not audio.is_file():
        return None, 'audio_missing'
    if file_digest(audio) != rec['audio_sha256']:
        return None, 'audio_bytes_changed'
    return {'clip_id':rec['clip_id'], 'audio_sha256':rec['audio_sha256'],
            'record_sha256':canonical_digest(rec), 'annotation_sha256':canonical_digest(annotation),
            'source_root_id':rec['source_root_id'], 'performed_words':annotation['performed_words'],
            'performed_taamim':annotation['performed_taamim'], 'label_source':annotation['label_source'],
            'reviewed_by':annotation['reviewed_by'], 'reviewed_at':annotation['reviewed_at']}, None


def validate_approval(path, rows, decisions, phrases_dir):
    path = Path(path)
    if not path.is_file():
        raise IntakeError('training approval manifest missing; run voice_intake.py --promote after human review')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(f'cannot read approval manifest: {exc}') from exc
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION or payload.get('kind') != APPROVAL_KIND:
        raise IntakeError('invalid approval schema; legacy allowed_clip_ids cannot approve training')
    entries = payload.get('entries')
    if not isinstance(entries, list):
        raise IntakeError('approval entries must be a list')
    if payload.get('manifest_sha256') != canonical_digest({k:v for k,v in payload.items() if k != 'manifest_sha256'}):
        raise IntakeError('approval manifest digest mismatch')
    indexed, seen, approved = {r['clip_id']:r for r in rows}, set(), []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('clip_id'), str):
            raise IntakeError('invalid approval entry')
        cid = entry['clip_id']
        if cid in seen or cid not in indexed:
            raise IntakeError('duplicate or absent approved clip')
        seen.add(cid)
        rec = indexed[cid]
        observed, problem = review_entry(rec, decisions.get(cid), phrases_dir)
        if problem:
            raise IntakeError(f'approved clip {cid} is no longer eligible: {problem}')
        if entry != observed:
            raise IntakeError(f'approved clip {cid} provenance or performed labels changed')
        approved.append((rec, decisions[cid], observed))
    return approved
