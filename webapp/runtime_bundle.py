"""Read-only, standard-library validation of an observed serving bundle.

Fingerprints establish file identity from observation onward. They do not
retroactively establish what a legacy checkpoint was trained against.
"""
import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import re
import sys

SCHEMA_VERSION = 1
STATUS = 'observed_binding_not_training_provenance'
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
REQUIRED_ROLES = frozenset({
    'cnn', 'verify_head', 'sequence', 'channel_stress', 'channel_dagesh',
    'channel_guttural', 'channel_sheva', 'calibration_global',
    'calibration_per_class', 'ctc_weights', 'ctc_config', 'ctc_vocab',
    'ctc_preprocessor', 'ctc_tokenizer', 'ctc_special_tokens',
    'frontend_server', 'frontend_training', 'frontend_verify',
    'frontend_sequence', 'frontend_alignment', 'frontend_ctc',
    'frontend_text_normalization', 'frontend_stress', 'frontend_sheva',
    'frontend_severity', 'frontend_channel_policy', 'frontend_capture',
    'frontend_practice', 'runtime_validator'})
ALLOWED_ROLES = REQUIRED_ROLES | {'ctc_added_tokens','calibration_channel_candidate'}


class RuntimeBundleError(ValueError):
    """Selected serving artifacts no longer match their observed bundle."""


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bundle_digest(payload):
    try:
        raw = json.dumps({k:v for k,v in payload.items() if k != 'bundle_sha256'},
                         sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    except (ValueError, TypeError) as exc:
        raise RuntimeBundleError('bundle contains unsupported values') from exc
    return hashlib.sha256(raw).hexdigest()


def library_versions():
    versions={'python':'.'.join(map(str,sys.version_info[:3]))}
    for name in ('numpy','torch','torchaudio','librosa','transformers','fastapi','soundfile'):
        try:
            versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name]=None
    return versions


def relative_file(root, value):
    """Reject absolute paths, traversal and symlink escapes from project root."""
    if not isinstance(value, str) or not value or '\\' in value or '\0' in value:
        raise RuntimeBundleError('artifact path must be a project-relative POSIX path')
    posix = PurePosixPath(value)
    if posix.is_absolute() or '..' in posix.parts or '.' in value.split('/') or str(posix) != value:
        raise RuntimeBundleError('artifact path escapes its project or is noncanonical')
    root = Path(root).resolve()
    path = root.joinpath(*posix.parts)
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise RuntimeBundleError('artifact symlink escapes project root') from exc
    if not path.is_file():
        raise RuntimeBundleError('required bundle artifact is missing')
    return path


def validate_bundle(payload, project_root, expected_options=None):
    """Validate all files before accepting traffic; returns the same manifest.

    expected_options must describe the active runtime selection. With no
    argument, only the existing uncalibrated-channel selection is accepted.
    """
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION:
        raise RuntimeBundleError('runtime bundle schema is unsupported')
    if payload.get('kind') != 'taamify.runtime-bundle' or payload.get('status') != STATUS:
        raise RuntimeBundleError('runtime bundle kind or evidence status is invalid')
    if payload.get('binding_evidence') != 'observed_files_only':
        raise RuntimeBundleError('legacy bundle must not claim verified training provenance')
    if not isinstance(payload.get('code_commit'),str) or not re.fullmatch(r'[0-9a-f]{40}',payload['code_commit']):
        raise RuntimeBundleError('runtime code commit is missing or invalid')
    for key in ('dirty_diff_sha256','dirty_status_sha256'):
        if not isinstance(payload.get(key),str) or not SHA256.fullmatch(payload[key]):
            raise RuntimeBundleError('runtime source provenance digest is missing or invalid')
    if not isinstance(payload.get('bundle_sha256'), str) or not SHA256.fullmatch(payload['bundle_sha256']):
        raise RuntimeBundleError('runtime bundle digest is invalid')
    if payload['bundle_sha256'] != bundle_digest(payload):
        raise RuntimeBundleError('runtime bundle digest mismatch')
    options = {'channel_calibration': None} if expected_options is None else expected_options
    if payload.get('runtime_options') != options:
        raise RuntimeBundleError('selected runtime options do not match bundle')
    if payload.get('library_versions') != library_versions():
        raise RuntimeBundleError('runtime library versions changed since bundle creation')
    files = payload.get('files')
    if not isinstance(files, list):
        raise RuntimeBundleError('runtime bundle files must be a list')
    indexed = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get('role'), str):
            raise RuntimeBundleError('runtime bundle artifact entry is invalid')
        role = entry['role']
        if role not in ALLOWED_ROLES:
            raise RuntimeBundleError('runtime bundle contains an unsupported role')
        if role in indexed:
            raise RuntimeBundleError('runtime bundle contains a duplicate role')
        indexed[role] = entry
        if not isinstance(entry.get('sha256'), str) or not SHA256.fullmatch(entry['sha256']):
            raise RuntimeBundleError(f'{role}: invalid artifact digest')
        if type(entry.get('bytes')) is not int or entry['bytes'] < 0:
            raise RuntimeBundleError(f'{role}: invalid artifact size')
        path = relative_file(project_root, entry.get('path'))
        if path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['sha256']:
            raise RuntimeBundleError(f'{role}: artifact bytes changed since bundle creation')
        if 'class_order' in entry and (not isinstance(entry['class_order'], list) or
                not entry['class_order'] or not all(isinstance(c,str) and c for c in entry['class_order']) or
                len(set(entry['class_order'])) != len(entry['class_order'])):
            raise RuntimeBundleError(f'{role}: invalid class order')
    if REQUIRED_ROLES - indexed.keys():
        raise RuntimeBundleError('runtime bundle is missing required role(s): ' +
                                 ', '.join(sorted(REQUIRED_ROLES - indexed.keys())))
    selected = options.get('channel_calibration')
    if selected is not None:
        if 'calibration_channel_candidate' not in indexed or indexed['calibration_channel_candidate']['path'] != selected:
            raise RuntimeBundleError('selected channel calibration is not bound')
    elif 'calibration_channel_candidate' in indexed:
        raise RuntimeBundleError('bundle binds a channel candidate that is not selected')
    for role in ('verify_head', 'sequence'):
        if indexed['cnn'].get('class_order') and indexed[role].get('class_order'):
            if indexed['cnn']['class_order'] != indexed[role]['class_order']:
                raise RuntimeBundleError(f'{role}: observed class order differs from CNN')
    if not isinstance(payload.get('frontend'), dict) or not payload['frontend']:
        raise RuntimeBundleError('runtime frontend description is missing')
    return payload


def load_bundle(path, project_root, expected_options=None):
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeBundleError('cannot read selected runtime bundle') from exc
    return validate_bundle(payload, project_root, expected_options)


def public_summary(payload):
    """Safe API summary: no local paths, input audio or identifying metadata."""
    if payload is None:
        return {'status':'unbound', 'bundle_sha256':None, 'roles':[]}
    return {'schema_version':payload['schema_version'],
            'bundle_sha256':payload['bundle_sha256'], 'status':payload['status'],
            'binding_evidence':payload['binding_evidence'],
            'roles':sorted(entry['role'] for entry in payload['files'])}
