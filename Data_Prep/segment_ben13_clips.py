"""
Segment Ben13's combination clips into per-ta'am audio pieces.
================================================================
Ben13's combination clips (e.g. "Mercha Tipcha Munach Etnachta" as one
~5-second clip) are the teacher chanting each ta'am's own name in sequence,
sung with that ta'am's melody — confirmed by transcribing sample clips with
mlx-whisper and matching the output phonetically against the manifest's own
`text`/`textSfaradi` fields (see Milestone 1C in the plan). That means the
exact expected text for forced alignment is already in the manifest — no
Sefaria lookup needed, unlike every other alignment in this project.

Naive cuts don't work: most of a clip's duration is melismatic tail that
Whisper's own word timestamps truncate and that carries no silence to cut
on (confirmed with ffmpeg silencedetect/volumedetect — the "quiet" tail is
still full-volume singing). `verse_align.align_verse()` is the right tool —
it's a plain (waveform, word-list) -> (spans, score) forced alignment with
no Sefaria dependency, already proven on real chanted Hebrew.

Its MIN_SCORE/MAX_EDGE_GAP thresholds are calibrated for long multi-word
Torah verses, not 1-5 word phrases, so this script records every score as a
quality weight rather than hard-gating on it — consistent with how
verse_align.py itself treats scores. Flag (don't discard) low-confidence
cuts for manual review, especially rare-vocabulary clips (Yerach Ben Yomo /
Qarney Para) where even Whisper's ASR struggled.

Output:
  data/ben13_segments/<type>/<Taam>_<type>-<order>.wav
  data/ben13_segments/<type>/manifest.json
    [{taam, source_type, source_order, source_url, ctc_score, duration,
      local_path}]

Usage:
  python Data_Prep/segment_ben13_clips.py             # all types
  python Data_Prep/segment_ben13_clips.py --types teamimtora
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import TAAM_MAP, normalize_word  # noqa: E402
from ctc_align import SR, load_recording  # noqa: E402
from verse_align import align_verse  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEN13_DIR = PROJECT_DIR / "data" / "ben13"
OUT_DIR = PROJECT_DIR / "data" / "ben13_segments"

TYPES = ["teamimtora", "teamim-meg", "teamim-haf"]

# verse_align.MIN_SCORE (-3.2) was calibrated on long verses; these are
# 1-5 word phrases, a different regime. Flag well below that as suspect
# rather than silently trusting it, without hard-discarding anything.
FLAG_SCORE = -3.5

# align_verse() places an internal word boundary at the first acoustic
# evidence of the NEXT word (see its own docstring). On these clips that
# evidence is usually a new phoneme, but a ta'am name's melismatic tail
# ("סלסול") sustains the same vowel with no new phoneme to lock onto, so
# CTC is starved of evidence right where it matters and the boundary lands
# early. That produces two symptoms that a manual review of all 114 cut
# segments confirmed are the dominant complaint pattern: word i's own tail
# truncated, and/or word i+1's segment opening on word i's leftover tail.
# Both are the same early boundary, so shifting it later fixes both at
# once. Capped at half of the next word's own span so a short word is
# never swallowed entirely.
#
# Calibration: the first attempt used 0.15s, which the reviewer's second
# listening pass showed overshoots -- nearly every re-reviewed clip came
# back "now I hear <my word> plus the start of the next one", with one
# note pinning it precisely ("delete the last tenth of a second and it
# would be good"). 0.05 keeps the melismatic tail without crossing into
# the next word's first consonant.
BOUNDARY_SHIFT_SEC = 0.05

# One combo entry's own source text is internally inconsistent: its first
# word is spelled "קדמא" (the letters for Qadma) but carries Pashta's own
# diacritic, and the manifest's taam_sequence agrees with the diacritic
# (Pashta) -- so the segmentation is self-consistent, but a listener
# expecting to hear "Qadma" hears this segment labeled Pashta instead.
# Confirmed identical across all three clip-type copies of this combo
# (teamimtora order 9, teamim-meg/teamim-haf order 90). This is a property
# of Ben13's own source text, not a timing bug -- not auto-fixable, so it's
# carried through to the review tool as a flag instead of silently
# corrected or silently trusted.
KNOWN_LABEL_NOTES = {
    ("teamimtora", 9, 0): (
        "Source text spells this word קדמא but tags it with Pashta's own "
        "diacritic, and taam_sequence agrees with the diacritic (Pashta) -- "
        "confirmed identical across all 3 clip-type copies (order 9/90/90). "
        "Confirmed by the reviewer: קדמא is this tradition's own colloquial "
        "name for Pashta, not a Ben13 source error -- not the same as the "
        "academically-distinct Qadma class (see order 140/Qadma->Geresh), "
        "which is a real, separate open question: does *that* class also "
        "go by קדמא colloquially, or something else? See sephardi_names.json."
    ),
    ("teamim-meg", 90, 0): "same as teamimtora order 9 -- see that note.",
    ("teamim-haf", 90, 0): "same as teamimtora order 9 -- see that note.",
}


def token_taams(token: str) -> list:
    """Ta'amim decoded from one token's own diacritics, in order."""
    return [TAAM_MAP[ch] for ch in token if ch in TAAM_MAP]


def clean_text(text: str) -> str:
    r"""Undo Ben13's whitespace escaping before tokenizing.

    The source text carries both real U+00A0 non-breaking spaces and the
    four *literal* characters \xA0. str.split() separates on the former but
    not the latter, so "מונח\xA0רביעי" arrives as one token holding two
    ta'amim -- which words_from_entry then correctly refuses to split,
    silently dropping the entry or pushing it onto the `text` fallback.
    Normalising here is what lets those entries tokenize at all."""
    if not text:
        return ""
    for bad in ("\\xA0", "\\xa0", "\xa0", "\\n"):
        text = text.replace(bad, " ")
    return text


def words_from_entry(entry: dict):
    """(words, taam_per_word, source_field) for forced alignment, or
    (None, None, None) if no field's tokenization matches taam_sequence 1:1.

    A token with zero ta'amim (e.g. "סוף-פסוק", the end-of-verse marker some
    clips end on) is kept as a real word for alignment — the audio still
    speaks it — but tagged with taam=None so no segment is emitted for it,
    which also stops it from silently swallowing the true final ta'am's span
    (align_verse extends each word to the onset of the next).

    `textSfaradi` is tried first and is the field to trust: it's the
    Sephardic name the teacher actually sings. Falling back to `text` (the
    academic/Ashkenazi naming) is a real hazard, confirmed by review --
    order 180's textSfaradi is the single word "רביע" while its `text` is
    the two words "מונח רביעי", and the audio sings only one. Aligning the
    two-word text onto one-word audio produced two garbage segments and the
    reviewer's "why are you cutting a one-word clip at all?" complaint. So
    the fallback still happens (some entries genuinely only have `text`) but
    is now reported, not silent: `source_field` is returned so the caller can
    flag it for manual attention."""
    seq = entry["taam_sequence"]
    for field in ("textSfaradi", "text"):
        raw = clean_text(entry.get(field) or "")
        tokens = raw.split()
        if not tokens:
            continue
        per_token = [token_taams(t) for t in tokens]
        if any(len(g) > 1 for g in per_token):
            continue  # a word carrying >1 ta'am can't be split by word alignment
        flat = [g[0] for g in per_token if g]
        if flat != seq:
            continue
        words = [{"raw": t, "norm": normalize_word(t)} for t in tokens]
        taams = [g[0] if g else None for g in per_token]
        return words, taams, field
    return None, None, None


def load_phantom_words() -> dict:
    """detect_phantom_words.py's output: words Ben13's text declares that
    its audio never sings, found by leave-one-out CTC hypothesis testing
    and validated 7/7 against the reviewer's hand-labelled clips.

    Dropping these before alignment is what stops the whole downstream
    cascade: a phantom word forces CTC to compress the real words to make
    room for it, shifting every boundary after it and mislabeling every
    segment past it."""
    path = OUT_DIR / "phantom_words.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_manual_cuts() -> dict:
    """Reviewer-corrected transcript + hand-placed boundaries per source clip
    (webapp/static/cut.html), keyed "{type}-{order}".

    These win over CTC unconditionally. Ben13's own text is not a reliable
    transcript of its own audio -- several clips list words that are simply
    never sung (order 20's "שופר-הולך", order 240's middle "מאריך", order
    190's leading "שופר-הולך") -- and a forced alignment given a word list
    longer than the audio cannot fail loudly: it silently compresses,
    shifting every boundary and mislabeling every segment after the missing
    word. No scoring threshold catches that, because the alignment is
    internally consistent; only a human listening to the clip does."""
    path = OUT_DIR / "manual_cuts.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=TYPES, choices=TYPES)
    ap.add_argument("--boundary-shift", type=float, default=BOUNDARY_SHIFT_SEC,
                    help="seconds to push each internal word boundary later "
                         "(default %(default)s, see BOUNDARY_SHIFT_SEC docs)")
    args = ap.parse_args()

    manual = load_manual_cuts()
    phantoms = load_phantom_words()
    manifest_out = []
    n_manual = n_phantom = 0
    for type_name in args.types:
        manifest_path = BEN13_DIR / type_name / "manifest.json"
        if not manifest_path.exists():
            print(f"  {type_name}: no manifest — run download_ben13.py first")
            continue
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
        combos = [e for e in entries if len(e["taam_sequence"]) > 1]
        print(f"\n{'=' * 70}\n  {type_name}: {len(combos)} combination clips\n{'=' * 70}")

        out_dir = OUT_DIR / type_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for e in sorted(combos, key=lambda x: x["order"]):
            seq = e["taam_sequence"]
            audio_path = PROJECT_DIR / e["local_path"]
            key = f"{type_name}-{e['order']}"

            # A hand-corrected cut replaces the whole CTC path: its word
            # list is the reviewer's transcript of what is actually sung,
            # which is the one thing forced alignment cannot recover on its
            # own when Ben13's text lists words the audio never says.
            if key in manual and audio_path.exists():
                y = load_recording(audio_path)
                for i, w in enumerate(manual[key].get("words", [])):
                    cls = w.get("taam")
                    if not cls:
                        continue
                    start, end = float(w["start"]), float(w["end"])
                    clip = y[int(start * SR):int(end * SR)]
                    if len(clip) < SR * 0.05:
                        continue
                    local_name = f"{cls}_{type_name}-{e['order']}_{i}.wav"
                    sf.write(out_dir / local_name, clip, SR)
                    manifest_out.append({
                        "taam": cls, "source_type": type_name,
                        "source_order": e["order"], "source_url": e["resolved_url"],
                        "ctc_score": None, "duration": round(len(clip) / SR, 3),
                        "start": round(start, 3), "end": round(end, 3),
                        "local_path": str((out_dir / local_name).relative_to(PROJECT_DIR)),
                        "low_confidence": False, "cut_source": "manual",
                        "word_text": w.get("label", ""),
                    })
                n_manual += 1
                print(f"    order {e['order']:>3} MANUAL  "
                      f"{' -> '.join(w.get('taam') or '-' for w in manual[key].get('words', []))}")
                continue

            words, taams, src_field = words_from_entry(e)

            # Drop a detected phantom word (and its ta'am) before aligning.
            ph = phantoms.get(key, {})
            dropped_word = None
            if words is not None and ph.get("phantom_index") is not None:
                pi = ph["phantom_index"]
                if 0 <= pi < len(words):
                    dropped_word = words[pi]["raw"]
                    words = words[:pi] + words[pi + 1:]
                    taams = taams[:pi] + taams[pi + 1:]
                    n_phantom += 1

            if words is None:
                print(f"    SKIP order {e['order']}: no clean 1:1 word/ta'am "
                      f"tokenization in text or textSfaradi — needs manual "
                      f"review ('{e['text']}')")
                continue

            if not audio_path.exists():
                print(f"    SKIP order {e['order']}: {audio_path.name} not on "
                      f"disk (download_ben13.py reported this one 404'd)")
                continue
            y = load_recording(audio_path)
            spans, score, _ = align_verse(y, words)
            if spans is None:
                print(f"    SKIP order {e['order']}: forced alignment failed "
                      f"('{e['text']}')")
                continue

            flag = " <- LOW SCORE, review by hand" if score < FLAG_SCORE else ""
            if src_field != "textSfaradi":
                flag += "  <- fell back to `text`, SUSPECT (see words_from_entry)"
            if dropped_word:
                flag += f"  <- dropped phantom word '{dropped_word}'"
            print(f"    order {e['order']:>3} score {score:>6.2f}{flag}  "
                  f"{' -> '.join(seq)}")

            for i in range(len(taams) - 1):
                if i not in spans or (i + 1) not in spans:
                    continue
                a_start, _a_end = spans[i]
                b_start, b_end = spans[i + 1]
                shift = min(args.boundary_shift, (b_end - b_start) * 0.5)
                new_boundary = b_start + shift
                spans[i] = (a_start, new_boundary)
                spans[i + 1] = (new_boundary, b_end)

            for i, cls in enumerate(taams):
                if cls is None:
                    continue
                start, end = spans[i]
                s_idx, e_idx = int(start * SR), int(end * SR)
                clip = y[s_idx:e_idx]
                if len(clip) < SR * 0.05:
                    continue
                # position i is part of the filename: a class can repeat
                # within one combo clip (e.g. "Mercha -> Tipecha -> Mercha"),
                # and without it the second occurrence silently overwrote the
                # first on disk -- confirmed after the fact via a 114-vs-102
                # unique-filename count mismatch against the manifest.
                local_name = f"{cls}_{type_name}-{e['order']}_{i}.wav"
                local_path = out_dir / local_name
                sf.write(local_path, clip, SR)
                entry = {
                    "taam": cls,
                    "source_type": type_name,
                    "source_order": e["order"],
                    "source_url": e["resolved_url"],
                    "ctc_score": round(score, 3),
                    "duration": round(len(clip) / SR, 3),
                    # kept so the manual cut editor can seed its handles from
                    # this alignment instead of making the reviewer place
                    # every boundary from scratch
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "local_path": str(local_path.relative_to(PROJECT_DIR)),
                    "low_confidence": score < FLAG_SCORE,
                    "cut_source": "ctc",
                    "text_fallback": src_field != "textSfaradi",
                    "word_text": words[i]["raw"],
                }
                label_note = KNOWN_LABEL_NOTES.get((type_name, e["order"], i))
                if label_note:
                    entry["label_note"] = label_note
                manifest_out.append(entry)

    out_manifest = OUT_DIR / "manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, ensure_ascii=False, indent=1)

    # lets the review tool warn when a saved note's audio has since been
    # re-cut (e.g. by this boundary-shift fix) and needs reconfirming.
    with open(OUT_DIR / "last_regenerated.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "boundary_shift": args.boundary_shift},
                  f)

    print(f"\n{'=' * 70}")
    by_class = {}
    for item in manifest_out:
        by_class.setdefault(item["taam"], []).append(item)
    for cls, items in sorted(by_class.items()):
        n_low = sum(1 for i in items if i["low_confidence"])
        print(f"  {cls:<16} {len(items)} segment(s)"
              f"{f' ({n_low} low-confidence)' if n_low else ''}")
    print(f"\nwrote {len(manifest_out)} segments -> {out_manifest.relative_to(PROJECT_DIR)}"
          f"  ({n_manual} from hand-corrected cuts, "
          f"{n_phantom} with a phantom word dropped)")


if __name__ == "__main__":
    main()
