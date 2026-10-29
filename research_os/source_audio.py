"""Resolve an exact Ben13 source key without silently choosing the last file.

The same key can be repeated in different aliyot. Path equality is not audio
identity: equivalent copies are accepted only after byte and text checks.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import unicodedata


class SourceConflict(ValueError):
    """The source provenance cannot be resolved without human intervention."""


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_key(row):
    return tuple(str(row[field]) for field in ('book', 'perek', 'pasuk', 'block'))


def canonical_words(text):
    from Data_Prep.taam_grammar import verse_tokens
    return tuple((unicodedata.normalize('NFD', word), taam)
                 for word, taam in verse_tokens(text))


class SourceResolver:
    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.by_key = defaultdict(list)
        for row in manifest:
            self.by_key[source_key(row)].append(row)
        self.counts = Counter()
        self.cache = {}
        self.hashes = {}

    def resolve(self, key):
        key = tuple(map(str, key))
        if key in self.cache:
            value = self.cache[key]
            if isinstance(value, Exception):
                raise value
            return value
        self.counts['keys_checked'] += 1
        rows = self.by_key.get(key, [])
        try:
            if not rows:
                raise SourceConflict('source_key_missing')
            paths = sorted({row['local_path'] for row in rows})
            self.counts['path_collision_keys_checked'] += len(paths) > 1
            texts = {canonical_words(row['text']) for row in rows}
            if len(texts) != 1:
                self.counts['text_conflict_keys'] += 1
                raise SourceConflict('source_text_conflict')
            if not next(iter(texts)):
                raise SourceConflict('source_text_empty')
            files = []
            for relative in paths:
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root):
                    raise SourceConflict('source_path_escapes_project')
                if not path.is_file():
                    raise SourceConflict('source_file_missing')
                if relative not in self.hashes:
                    self.hashes[relative] = sha256_file(path)
                files.append((relative, self.hashes[relative]))
            if len({value for _, value in files}) != 1:
                self.counts['audio_content_conflict_keys'] += 1
                raise SourceConflict('source_audio_content_conflict')
            if len(files) > 1:
                self.counts['byte_identical_path_collision_keys'] += 1
            text = sorted(rows, key=lambda r: (r['local_path'], str(r.get('id', ''))))[0]['text']
            value = {'key': key, 'path': str(self.root / paths[0]),
                     'relative_paths': paths, 'source_audio_sha256': files[0][1],
                     'source_root_id': 'ben13-audio:' + files[0][1],
                     'text': text, 'tokenization': next(iter(texts)),
                     'aliases': [{'id': str(r.get('id', '')), 'aliya': str(r.get('aliya', '')),
                                  'local_path': r['local_path']} for r in rows]}
            self.cache[key] = value
            self.counts['resolved_keys'] += 1
            return value
        except (SourceConflict, KeyError, ValueError) as exc:
            error = exc if isinstance(exc, SourceConflict) else SourceConflict('invalid_source_record')
            self.cache[key] = error
            self.counts['quarantined_keys'] += 1
            raise error

    def report(self):
        return {'manifest_unique_keys': len(self.by_key),
                'manifest_path_collision_keys': sum(len({r['local_path'] for r in rows}) > 1
                                                    for rows in self.by_key.values()),
                'selected_key_checks': dict(self.counts)}
