"""
Verse-time localization when Whisper's word-level ASR cannot be trusted.
=========================================================================
Gate 2 on reader R7 (Yemenite nusach): Whisper's median word confidence there
is 0.593, against 0.973 on R1. Sampled transcripts confirm it, not just the
number — "ברא" comes out "ברו", extended passages degenerate into repeated
nonsense ("granting ע granting ע..."), the classic signature of an ASR model
hearing phonology (Yemenite ayin, chet, qof; different vowel qualities) it was
never trained on. clean_align_v2's word-level survival on R7 is 9.0% against
41% on R1/R2, and the shortfall is almost entirely "dropped_unmatched" — the
transcript does not textually resemble the canonical text often enough for
verse_align's macro-window (built from surviving word timestamps) to span a
verse at all.

But the forced-alignment CTC model is a separate network from Whisper, and a
15-second, no-Whisper-timing test on R7's Bereshit scored -2.61 — worse than
R1/R2's -1.9ish, but inside the -3.2 threshold already in use, and the returned
per-word spans were plausible. The acoustic model tolerates this material far
better than Whisper's transcription does. The bottleneck is macro-localization,
not the forced alignment itself.

This replaces Whisper-derived macro windows with proportional ones: a verse's
time window is estimated from its share of the parasha's total canonical
character count, against the recording's total duration — no ASR transcription
involved at all, just "this verse is nine characters out of two thousand, so it
gets roughly that share of the runtime." Real cantillation is not perfectly
uniform in tempo, so the window is padded generously and only the forced
alignment's own score decides whether the result is kept.

Usage:
  python Data_Prep/proportional_verse_locate.py --reader R7 --parasha 01_bereshit
  python Data_Prep/proportional_verse_locate.py --reader R7 --all
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import PARASHA_BOUNDS, extract_taamim  # noqa: E402
from verse_align import verse_words, align_verse, LAG_START, LAG_END, MIN_SCORE  # noqa: E402
from ctc_align import load_recording, SR  # noqa: E402
from stressed_syllable import stressed_syllable  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
PARTS = Path.home() / ".cache" / "taamify" / "proportional_parts"

# Real chanting is not metronomic — a verse can run well over its proportional
# share, especially one carrying a long melisma. Generous relative padding
# rather than a fixed number of seconds, since parshiyot range from a few
# minutes to nearly an hour.
REL_PAD = 0.35
MIN_WINDOW_SEC = 1.5
MAX_WINDOW_SEC = 60.0

# A word-level sanity bound. The verse-initial word of a parasha has nothing
# bounding it on the left but the window edge, so if the recording opens with
# a few seconds of non-Torah content (a blessing, an announcement — audible in
# R7's Whisper transcript before the text proper), that word can absorb all of
# it. extract_features_v3 would silently truncate an over-long slice to its
# first ~2s rather than reject it, so the filter belongs here, at the source.
MIN_WORD_SEC = 0.08
MAX_WORD_SEC = 4.0


def verse_words_tagged(book: str, chapter: int, verse: int) -> list:
    """verse_words, plus the ta'am each token carries — needed for training
    labels, not just for forced alignment. Silluq is encoded with the meteg
    codepoint rather than a cantillation mark, so it is assigned positionally
    to the verse-final word, exactly as clean_align_v2.build_sefaria_words does."""
    words = verse_words(book, chapter, verse)
    for w in words:
        taamim = extract_taamim(w["raw"])
        w["taamim"] = taamim
        w["taam"] = taamim[-1] if taamim else None
    if words and words[-1]["taam"] is None:
        words[-1]["taam"] = "Siluk"
        words[-1]["taamim"] = words[-1]["taamim"] + ["Siluk"]
    return words


def parasha_verses(parasha: str):
    """[(book, chapter, verse, char_len)] for every verse in the parasha,
    char_len from the canonical (unpointed) letter count."""
    book, sc, sv, ec, ev = PARASHA_BOUNDS[parasha]
    out = []
    for ch in range(sc, ec + 1):
        v_from = sv if ch == sc else 1
        # verse_words returns [] past the chapter end, so probe upward until it does
        v = v_from
        while True:
            words = verse_words(book, ch, v)
            if not words:
                break
            if ch == ec and v > ev:
                break
            out.append((book, ch, v, sum(len(w["norm"]) for w in words)))
            v += 1
    return out


def proportional_windows(parasha: str, duration: float):
    """{(book, ch, v): (start, end)} spanning the whole recording by character share."""
    verses = parasha_verses(parasha)
    total_chars = sum(c for *_, c in verses) or 1
    out, cursor = {}, 0.0
    for book, ch, v, c in verses:
        share = duration * c / total_chars
        out[(book, ch, v)] = (cursor, cursor + share)
        cursor += share
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--parasha", help="single parasha key, e.g. 01_bereshit")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        parshiyot = sorted(PARASHA_BOUNDS)
    elif args.parasha:
        parshiyot = [args.parasha]
    else:
        raise SystemExit("give --parasha or --all")

    PARTS.mkdir(parents=True, exist_ok=True)
    written, stats = {}, Counter()

    for n, parasha in enumerate(parshiyot, 1):
        wav = PROJECT_DIR / "data" / args.reader / "wav" / f"{parasha}.wav"
        if not wav.exists():
            continue
        part = PARTS / f"{args.reader}__{parasha}.json"
        if part.exists():
            with open(part, encoding="utf-8") as f:
                cached = json.load(f)
            written.update(cached["windows"])
            stats.update(cached["stats"])
            print(f"[{n}/{len(parshiyot)}] {parasha} (cached)", flush=True)
            continue

        t0 = time.time()
        y_full = load_recording(wav)
        duration = len(y_full) / SR
        windows = proportional_windows(parasha, duration)

        part_windows, part_stats = {}, Counter()
        for (book, ch, v), (lo, hi) in windows.items():
            lo = max(0.0, lo - REL_PAD * (hi - lo))
            hi = min(duration, hi + REL_PAD * (hi - lo))
            if not (MIN_WINDOW_SEC < hi - lo <= MAX_WINDOW_SEC):
                part_stats["bad_window"] += 1
                continue
            words = verse_words_tagged(book, ch, v)
            if not words:
                continue
            y = y_full[int(lo * SR):int(hi * SR)]
            spans, score, _ = align_verse(y, words)
            if spans is None:
                part_stats["align_failed"] += 1
                continue
            if score < MIN_SCORE:
                part_stats["low_score"] += 1
                continue
            for w_i, w in enumerate(words):
                if w_i not in spans:
                    continue
                w_start = lo + spans[w_i][0] - LAG_START
                w_end = lo + spans[w_i][1] - LAG_END
                if not (MIN_WORD_SEC <= w_end - w_start <= MAX_WORD_SEC):
                    part_stats["bad_word_duration"] += 1
                    continue
                syl = stressed_syllable(w["raw"])
                key = f"{args.reader}|{parasha}|verse|{ch}:{v}|{w_i}"
                if w["taam"] is None:
                    part_stats["no_taam"] += 1
                    continue
                entry = {"word_clean": w["norm"], "word_raw": w["raw"],
                         "taam": w["taam"], "taamim_all": w["taamim"],
                         "verse_ref": f"{book} {ch}:{v}",
                         "word_start": round(w_start, 3), "word_end": round(w_end, 3),
                         "score": round(score, 3)}
                if syl.get("ok"):
                    n_let = max(syl["n_letters"], 1)
                    dur = w_end - w_start
                    entry["start"] = round(w_start + dur * syl["letter_start"] / n_let, 3)
                    entry["end"] = round(w_start + dur * syl["letter_end"] / n_let, 3)
                part_windows[key] = entry
                part_stats["aligned"] += 1

        with open(part, "w", encoding="utf-8") as f:
            json.dump({"windows": part_windows, "stats": dict(part_stats)}, f)
        written.update(part_windows)
        stats.update(part_stats)
        print(f"[{n}/{len(parshiyot)}] {parasha}: {part_stats['aligned']} aligned "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\naligned {stats['aligned']:,} words")
    for k, v in stats.most_common():
        if k != "aligned":
            print(f"  {k:<16}{v:>8,}")

    out = PROJECT_DIR / f"proportional_windows_{args.reader}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(written, f, ensure_ascii=False)
    print(f"\nwrote {len(written):,} words -> {out.name}")


if __name__ == "__main__":
    main()
