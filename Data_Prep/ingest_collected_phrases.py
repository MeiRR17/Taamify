"""Create a NEW feature snapshot from hash-bound human-reviewed performances.

Refuses absent/stale approvals before loading an aligner or writing anything.
Expected taamim and model statuses never supply training labels. Changed or
omitted words remain quarantined until separately reviewed and aligned.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

try:
    from .intake_contract import (IntakeError, file_digest, load_decisions,
                                  load_records, validate_approval)
except ImportError:
    try:
        from Data_Prep.intake_contract import (IntakeError, file_digest, load_decisions,
                                              load_records, validate_approval)
    except ModuleNotFoundError:
        from intake_contract import (IntakeError, file_digest, load_decisions,
                                     load_records, validate_approval)

ROOT = Path(__file__).resolve().parent.parent
PHRASES = ROOT/'app_assets/collected_phrases'


def runtime_tools():
    """Delay heavyweight imports until every approved source passed validation."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import librosa
    import numpy as np
    from clean_align_v2 import normalize_word
    from verse_align import align_verse
    from ctc_align import SR

    def load_audio(path):
        return librosa.load(str(path), sr=SR)[0]

    def features(y, start, end):
        s, e = max(0, int((start-.05)*SR)), min(len(y), int((end+.05)*SR))
        if e <= s or e-s < SR*.1:
            return None
        mel = librosa.feature.melspectrogram(y=y[s:e], sr=SR, n_mels=128, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)
        if db.shape[1] < 87:
            db = np.pad(db, ((0,0),(0,87-db.shape[1])), constant_values=-80.)
        return db[:,:87].astype(np.float32)
    return load_audio, align_verse, normalize_word, features


def create_snapshot(approved, phrases_dir, out_dir, approval_path, min_score, tools=None):
    """Write one immutable snapshot; staging prevents partial training outputs."""
    import numpy as np
    if not approved:
        return {'approved_clips':0, 'written_words':0, 'rejections':{}}
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise IntakeError('output directory exists; choose a new feature snapshot')
    load_audio, align, normalize_word, features = tools or runtime_tools()
    by_reader = defaultdict(lambda: ([], []))
    reasons, accepted_clips = Counter(), 0
    approval_digest = file_digest(approval_path)
    for rec, annotation, entry in approved:
        audio_path = Path(phrases_dir)/f"{rec['clip_id']}.wav"
        if file_digest(audio_path) != entry['audio_sha256']:
            raise IntakeError('audio changed after approval validation')
        y = load_audio(audio_path)
        words = annotation['performed_words']
        prepared = [{'raw':w, 'norm':normalize_word(w)} for w in words]
        spans, score, _ = align(y, prepared)
        if spans is None or score is None or not np.isfinite(score) or score < min_score:
            reasons['performed_alignment_failed_or_weak'] += 1
            continue
        reader = 'USER_' + hashlib.sha256(rec['voice_id'].encode()).hexdigest()[:12]
        specs, meta = by_reader[reader]
        staged = []
        for index, taam in enumerate(annotation['performed_taamim']):
            if taam is None:
                continue
            if index not in spans:
                reasons['performed_word_unaligned'] += 1
                continue
            start, end = spans[index]
            if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= len(y)/16000 + 1e-6):
                raise IntakeError('invalid performed-word boundaries')
            spec = features(y, start, end)
            if spec is None:
                reasons['feature_too_short'] += 1
                continue
            if spec.shape != (128,87) or not np.isfinite(spec).all():
                raise IntakeError('invalid performed-word feature shape or values')
            staged.append((spec, {
                'sample_id':f"{entry['audio_sha256']}:{index}",
                'source_root_id':rec['source_root_id'], 'audio_sha256':entry['audio_sha256'],
                'session_id':rec['session_id'], 'tradition':rec['tradition'],
                'taam':taam, 'taam_expected':rec['taamim'][index],
                'word_clean':prepared[index]['norm'], 'word_raw':words[index],
                'reader':reader, 'parasha':'user', 'verse_ref':rec['clip_id'],
                'segment_block':rec['clip_id'], 'start':float(start), 'end':float(end),
                'window':'word', 'align':'human_reviewed_phrase_ctc', 'ctc_score':float(score),
                'label_source':entry['label_source'], 'annotation_sha256':entry['annotation_sha256'],
                'capture_record_sha256':entry['record_sha256'], 'approval_manifest_sha256':approval_digest}))
        if staged:
            accepted_clips += 1
            for spec, record in staged:
                specs.append(spec)
                meta.append(record)
    written = sum(len(rows[1]) for rows in by_reader.values())
    summary = {'approved_clips':len(approved), 'accepted_clips':accepted_clips,
               'written_words':written, 'rejections':dict(reasons),
               'approval_manifest_sha256':approval_digest,
               'policy':'human-performed-labels-only', 'schema_version':2}
    if not written:
        return summary
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{out_dir.name}.', dir=out_dir.parent))
    try:
        for reader, (specs, meta) in sorted(by_reader.items()):
            if specs:
                np.save(staging/f'{reader}_X.npy', np.stack(specs))
                (staging/f'{reader}_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
        (staging/'intake_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        staging.rename(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phrases-dir', type=Path, default=PHRASES)
    ap.add_argument('--log', type=Path)
    ap.add_argument('--decisions', type=Path)
    ap.add_argument('--approval', type=Path)
    ap.add_argument('--out-dir', type=Path, help='NEW feature directory; required except validation-only')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--min-score', type=float, default=-2.5)
    args = ap.parse_args(argv)
    if not args.validate_only and args.out_dir is None:
        ap.error('--out-dir is required to preserve existing corpora')
    if not (-100 < args.min_score <= 0):
        ap.error('--min-score must be a finite negative log-probability')
    try:
        approval = args.approval or args.phrases_dir/'training_safe.json'
        if not approval.is_file():
            raise IntakeError('training approval manifest missing; no training output created')
        rows = load_records(args.log or args.phrases_dir/'phrases_log.jsonl')
        decisions = load_decisions(args.decisions or args.phrases_dir/'intake_decisions.json')
        approved = validate_approval(approval, rows, decisions, args.phrases_dir)
        summary = ({'approved_clips':len(approved), 'validation_only':True} if args.validate_only else
            create_snapshot(approved, args.phrases_dir, args.out_dir, approval, args.min_score))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (IntakeError, OSError, ValueError, TypeError) as exc:
        ap.exit(2, f'ingest refused: {exc}\n')


if __name__ == '__main__':
    main()
