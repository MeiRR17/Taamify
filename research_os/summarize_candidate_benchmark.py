"""Strict secondary scoring of immutable canonical-recognition observations.

Counts inserted words/accents, preserves all reference denominators, and never
labels the expected-text alignment API a free word recognizer. No inference.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import shutil

from Data_Prep.clean_align_v2 import TAAM_MAP
from research_os.source_audio import sha256_file


def consonants(word):
    return ''.join(c for c in word if '\u05d0' <= c <= '\u05ea')


def predicted_tokens(text):
    """One token per spoken-word hypothesis, preserving verse-final Siluk.

    Doubled same-name glyphs encode one word accent event. Distinct accents
    on one predicted word all count, rather than silently taking the last.
    """
    chunks = [c for c in re.split(r'(\s+|־|׃)', text) if c and not c.isspace() and c != '־']
    result = []
    for i, chunk in enumerate(chunks):
        if not consonants(chunk):
            continue
        marks = [TAAM_MAP[c] for c in chunk if c in TAAM_MAP]
        verse_final = i == len(chunks) - 1 or chunks[i + 1] == '׃'
        if not marks and '\u05bd' in chunk and verse_final:
            marks.append('Siluk')
        result.append({'word': chunk, 'consonants': consonants(chunk),
                       'taamim': list(dict.fromkeys(marks)), 'accent_glyphs': len(marks)})
    return result


def edit_alignment(reference, predicted):
    """Unit-cost orthographic word alignment; deterministic diagonal-first ties."""
    a, b = list(map(consonants, reference)), list(map(consonants, predicted))
    costs = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        costs[i][0] = i
    for j in range(len(b) + 1):
        costs[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    pairs, i, j = [], len(a), len(b)
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            i -= 1; j -= 1; pairs.append((i, j))
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1; pairs.append((i, None))
        else:
            j -= 1; pairs.append((None, j))
    return list(reversed(pairs))


def transcription_metrics(words, taamim, text):
    if len(words) != len(taamim):
        raise ValueError('reference word/accent lengths differ')
    tokens = predicted_tokens(text)
    counts = Counter(phrases=1, reference_words=len(words),
                     reference_taam_events=sum(t is not None for t in taamim),
                     predicted_words=len(tokens),
                     predicted_taam_events=sum(len(t['taamim']) for t in tokens),
                     predicted_accent_glyphs=sum(t['accent_glyphs'] for t in tokens),
                     true_positive_events=0, substitutions=0, deletions=0, insertions=0,
                     matched_words=0, whole_phrase_exact=0)
    pairs = edit_alignment(words, [t['word'] for t in tokens])
    for i, j in pairs:
        if i is None:
            counts['insertions'] += 1
        elif j is None:
            counts['deletions'] += 1
        else:
            same = consonants(words[i]) == tokens[j]['consonants']
            counts['matched_words'] += same
            counts['substitutions'] += not same
            counts['true_positive_events'] += bool(same and taamim[i] is not None
                                                   and taamim[i] in tokens[j]['taamim'])
    counts['false_positive_events'] = counts['predicted_taam_events'] - counts['true_positive_events']
    counts['false_negative_events'] = counts['reference_taam_events'] - counts['true_positive_events']
    counts['whole_phrase_exact'] = int(len(words) == len(tokens) and all(
        consonants(word) == token['consonants'] and token['taamim'] == ([taam] if taam else [])
        for word, taam, token in zip(words, taamim, tokens)))
    return counts


def api_metrics(taamim, response):
    results = response.get('words') or []
    checked = response.get('aligned') is True
    counts = Counter(phrases=1, reference_taam_events=sum(t is not None for t in taamim),
                     correct_taam_at_expected_position=0, alignment_rejected=int(not checked),
                     missing_heard_positions=0, whole_phrase_taam_exact=0)
    for i, taam in enumerate(taamim):
        if taam is None:
            continue
        heard = results[i].get('heard') if checked and i < len(results) else None
        counts['correct_taam_at_expected_position'] += heard == taam
        counts['missing_heard_positions'] += heard is None
    counts['whole_phrase_taam_exact'] = int(checked and len(results) == len(taamim)
        and all(t is None or results[i].get('heard') == t for i, t in enumerate(taamim)))
    return counts


def finalized(counts, external=True):
    result = dict(counts)
    if external:
        tp, fp, fn = (counts[k] for k in ('true_positive_events', 'false_positive_events', 'false_negative_events'))
        result.update(event_precision=tp / (tp + fp) if tp + fp else None,
                      event_recall=tp / (tp + fn) if tp + fn else None,
                      event_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                      word_error_rate=(counts['substitutions'] + counts['deletions'] + counts['insertions']) / counts['reference_words'] if counts['reference_words'] else None,
                      whole_phrase_exact_rate=counts['whole_phrase_exact'] / counts['phrases'] if counts['phrases'] else None)
    else:
        result.update(taam_at_expected_position_rate=counts['correct_taam_at_expected_position'] / counts['reference_taam_events'] if counts['reference_taam_events'] else None,
                      whole_phrase_taam_exact_rate=counts['whole_phrase_taam_exact'] / counts['phrases'] if counts['phrases'] else None)
    return result


def summarize(manifest_path, observations_path, original_summary_path, output):
    manifest_path, observations_path, original_summary_path, output = map(Path,
        (manifest_path, observations_path, original_summary_path, output))
    if output.exists():
        raise FileExistsError('summary output already exists')
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    observations = json.loads(observations_path.read_text())
    original = json.loads(original_summary_path.read_text())
    if original['manifest_sha256'] != sha256_file(manifest_path):
        raise ValueError('original run does not bind to this manifest')
    indexed = {row['sample_id']: row for row in observations}
    if len(indexed) != len(observations) or set(indexed) != {r['sample_id'] for r in manifest}:
        raise ValueError('duplicate/missing/extra observation IDs')
    output.mkdir(parents=True)
    (output / 'audio').mkdir()
    total_external, total_api, paired = Counter(), Counter(), []
    for reference in manifest:
        observation = indexed[reference['sample_id']]
        if observation['audio_sha256'] != reference['audio_sha256'] or sha256_file(reference['path']) != reference['audio_sha256']:
            raise ValueError('audio provenance mismatch')
        external = transcription_metrics(reference['words'], reference['taamim'], observation['external_transcription'])
        api = api_metrics(reference['taamim'], observation['api_result'])
        total_external.update(external); total_api.update(api)
        relative = 'audio/' + reference['sample_id'] + '.wav'
        shutil.copyfile(reference['path'], output / relative)
        paired.append({'sample_id': reference['sample_id'], 'parasha': reference['parasha'],
                       'verse': reference['verse'], 'block': reference['block'],
                       'audio_relative_path': relative, 'audio_sha256': reference['audio_sha256'],
                       'expected_words': reference['words'], 'expected_taamim': reference['taamim'],
                       'api_words': observation['api_result'].get('words') or [],
                       'api_aligned': observation['api_result'].get('aligned') is True,
                       'api_metrics': finalized(api, False),
                       'external_transcription': observation['external_transcription'],
                       'external_metrics': finalized(external),
                       'requires_human_reference_audio_check': bool(external['insertions'] or external['deletions'] or external['substitutions'])})
    report = {'schema_version': 1, 'status': 'strict_secondary_canonical_diagnostic',
              'phrases': len(manifest), 'external': finalized(total_external),
              'expected_text_alignment_api': finalized(total_api, False),
              'original_summary_totals_preserved': original['totals'],
              'external_model': original['external_model'],
              'input_hashes': {'manifest': sha256_file(manifest_path), 'observations': sha256_file(observations_path),
                               'original_summary': sha256_file(original_summary_path), 'scorer': sha256_file(__file__)},
              'metric_definitions': {
                  'external_joint_event': 'Correct aligned consonantal word spelling AND accent class; insertions count as false-positive events.',
                  'word_normalization': 'Hebrew letters only, including final forms; vowel/cantillation marks and punctuation removed. Plene spelling differences remain word substitutions.',
                  'alignment': 'Unit-cost word Levenshtein; deterministic diagonal-first ties, no accent-based oracle alignment.',
                  'accent_event': 'One event per distinct accent class on a predicted word; doubled identical glyphs do not create two semantic accents. Intermediate sof-pasuq Siluk is preserved.',
                  'api': 'Accent identity at supplied expected word position. This API receives expected words for forced alignment and is not evaluated as a free word recognizer.',
                  'denominators': 'All manifest references and all 40 phrases retained, including alignment rejection.'},
              'limitations': original['limitations'] + [
                  'These two metric families describe different interfaces and do not establish model superiority.',
                  'Reference mismatches require listening/human adjudication; extra output is not independently proven hallucination.',
                  'No learner-error precision, calibration, new-speaker generalization, or product readiness claim.']}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'paired_rows.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2) + '\n')
    (output / 'index.html').write_text(report_html(report, paired), encoding='utf-8')
    return report


def report_html(report, rows):
    e, a = report['external'], report['expected_text_alignment_api']
    esc = lambda value: html.escape(str(value))
    percent = lambda value: '—' if value is None else f'{100 * value:.2f}%'
    parts = ['''<!doctype html><html dir="rtl" lang="he"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Taamify — השוואת הקלטות</title><style>body{font:17px system-ui;max-width:1100px;margin:25px auto;padding:18px;background:#f5f7fa;color:#162234}section,details{background:white;border:1px solid #dce3eb;border-radius:12px;padding:18px;margin:16px 0}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{text-align:right;border-bottom:1px solid #e1e6ed;padding:9px;vertical-align:top}.flag{border-right:6px solid #c37300}.muted{color:#536579}audio{width:100%;margin:12px 0}.hebrew{font-size:24px;line-height:1.8}.id{direction:ltr;unicode-bidi:embed;font-size:12px;word-break:break-all}summary{cursor:pointer;font-weight:700}code{direction:ltr;unicode-bidi:embed}</style><h1>מה השתנה כשסופרים גם פלט עודף?</h1><p>בדיקת אבחון מקומית על 40 משפטים של בן13: קול יחיד, ציפייה טקסטואלית, ללא תיוג ביצוע אנושי עצמאי.</p><p class="muted">חפיפה אפשרית בין האודיו לאימון המודל החיצוני. אין כאן הוכחת גילוי טעויות לומד או הכללה לקורא חדש. שני הממשקים נבדקים באופן שונה ולכן אין להסיק מהטבלה לבדה עליונות מוצרית.</p>''']
    parts.append('<section><h2>המודל החיצוני: מילה וטעם יחד, כולל הוספות</h2><table>')
    for label, value in [('מילות מקור', e['reference_words']), ('מילים בפלט', e['predicted_words']),
                         ('אירועי טעם צפויים', e['reference_taam_events']), ('אירועי טעם בפלט', e['predicted_taam_events']),
                         ('התאמות משותפות מילה וטעם', e['true_positive_events']),
                         ('דיוק אירועים — precision', percent(e['event_precision'])),
                         ('כיסוי אירועי המקור — recall', percent(e['event_recall'])), ('F1 של אירועים', percent(e['event_f1'])),
                         ('שגיאת מילים — WER', percent(e['word_error_rate'])),
                         ('הוספות / החלפות / השמטות', f"{e['insertions']} / {e['substitutions']} / {e['deletions']}"),
                         ('משפט שלם תואם בדיוק', f"{e['whole_phrase_exact']} / {e['phrases']}")]:
        parts.append(f'<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>')
    parts.append('</table><p>מדד המילים בודק איות עיצורי מדויק; למשל כתיב מלא מול חסר נספר כהחלפה. סימני ניקוד אינם נבדקים כאן. כל טעם שהופיע במילה שנוספה נספר גם הוא.</p></section>')
    parts.append(f'''<section><h2>ה־API המקומי: טעם במיקום מילה שסופק מראש</h2><p>הממשק מקבל את המילים המצופות לצורך יישור. אין כאן מדידת זיהוי מילים חופשי.</p><p>טעם תואם במיקום המצופה: <b>{a['correct_taam_at_expected_position']} / {a['reference_taam_events']} ({percent(a['taam_at_expected_position_rate'])})</b>. משפטים עם כל הטעמים תואמים: <b>{a['whole_phrase_taam_exact']} / {a['phrases']}</b>. דחיות יישור: <b>{a['alignment_rejected']} / {a['phrases']}</b>. כל הדחיות נכללו במכנה.</p></section><h2>כל 40 ההקלטות — בלי סינון הצלחות</h2>''')
    for row in rows:
        flagged = row['requires_human_reference_audio_check'] or not row['api_aligned']
        parts.append(f'<details class="{"flag" if flagged else ""}" {"open" if flagged else ""}><summary>פרשה {esc(row["parasha"])} · פסוק {esc(row["verse"])} · בלוק {esc(row["block"])}{" · נדרשת בדיקה" if flagged else ""}</summary>')
        parts.append(f'<p class="id">{esc(row["sample_id"])}</p><audio controls preload="none" src="{esc(row["audio_relative_path"])}"></audio>')
        parts.append(f'<p>טקסט מקור מצופה</p><p class="hebrew">{esc(" ".join(row["expected_words"]))}</p><p>פלט המודל החיצוני</p><p class="hebrew">{esc(row["external_transcription"])}</p>')
        if row['requires_human_reference_audio_check']:
            em = row['external_metrics']
            parts.append(f'<p><b>אי־התאמה למקור:</b> {em["insertions"]} מילים נוספו, {em["substitutions"]} הוחלפו ו־{em["deletions"]} הושמטו. יש להאזין ולבדוק את התאמת קובץ המקור; זו אינה קביעה שהמודל הזָה.</p>')
        if not row['api_aligned']:
            parts.append('<p>ה־API דחה את היישור. כל מילות המקור נשארו במכנה.</p>')
        parts.append('<table><tr><th>מילה מצופה</th><th>טעם מצופה</th><th>טעם ששמע ה־API</th><th>הכרעת ה־API</th></tr>')
        for index, (word, expected) in enumerate(zip(row['expected_words'], row['expected_taamim'])):
            got = row['api_words'][index] if index < len(row['api_words']) else {}
            parts.append(f'<tr><td class="hebrew">{esc(word)}</td><td>{esc(expected)}</td><td>{esc(got.get("heard", "לא נבדק"))}</td><td>{esc(got.get("status", "לא נבדק"))}</td></tr>')
        parts.append('</table></details>')
    parts.append('<p class="muted">תוצאות הגלם נשמרו ללא שינוי. קובצי summary.json ו־paired_rows.json כוללים הגדרות, מונים וטביעות קלט לשחזור.</p></html>')
    return ''.join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--observations', type=Path, required=True)
    ap.add_argument('--original-summary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(summarize(args.manifest, args.observations, args.original_summary, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
