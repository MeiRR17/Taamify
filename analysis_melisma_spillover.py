"""
Does a ta'am's melody stay inside the stressed syllable?
========================================================
The windowing in extract_features_v3 assumes the ta'am is realized on the
stressed syllable. That is safe for the short conjunctives, but Pazer, Telisha,
Zarqa and Shalshelet are melismas, and if their melody runs past the syllable
then the stressed window truncates exactly the ta'amim the model already fails
on. This measures it instead of assuming either way.

The quantity is *melodic activity* — the rate of pitch change in semitones per
second, a(t) = |d/dt semitones(f0(t))| — and the statistic is the fraction of a
word's total melodic activity that falls inside its stressed syllable.

Energy (RMS) is deliberately not used: loudness peaks on the stressed syllable
by definition, so it would confirm the hypothesis for every ta'am regardless of
melody. Plain f0 variance is not used either: variance is order-blind, and a
slow glide between two notes can match a rapid melisma.

Controls:
  - pitch in semitones relative to each reader's median f0, never in Hz
  - octave jumps from the tracker are clipped, not measured as melody
  - phrase position is held constant: Siluk and Etnachta words are phrase-final
    and lengthen for prosodic reasons that have nothing to do with their ta'am,
    so disjunctives and conjunctives are compared separately
  - gold word boundaries only (R6); a Whisper boundary makes "spillover past the
    word edge" indistinguishable from an alignment error
  - a null model with the syllable boundary shuffled inside the word, to show
    the real boundary concentrates melody better than an arbitrary one

Usage:
  python analysis_melisma_spillover.py --limit 4000
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "Data_Prep"))

PROJECT_DIR = Path(__file__).resolve().parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
WINDOWS = PROJECT_DIR / "stressed_windows.json"

SR = 22050
FMIN, FMAX = 65.0, 600.0        # human singing range; keeps pyin off harmonics
MAX_SEMITONE_JUMP = 6.0         # per frame: beyond this is a tracker octave slip

# Disjunctives (mafsikim) end a phrase and lengthen; conjunctives (mesharetim)
# do not. Mixing them would measure prosodic finality, not cantillation.
DISJUNCTIVE = {
    "Siluk", "Etnachta", "Segol", "Shalshelet", "Zaqef_Qatan", "Zaqef_Gadol",
    "Tipecha", "Revia", "Zarqa", "Pashta", "Yetiv", "Tevir", "Geresh",
    "Geresh_Muqdam", "Gershayim", "Pazer", "Qarney_Para", "Telisha_Gedola",
}


def semitone_track(y, sr=SR):
    """f0 as semitones relative to the clip's own median, plus a voiced mask."""
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr,
                                 frame_length=1024, hop_length=256)
    if f0 is None or not np.any(voiced):
        return None, None, None
    ok = np.isfinite(f0) & voiced
    if ok.sum() < 4:
        return None, None, None
    ref = np.nanmedian(f0[ok])
    if not np.isfinite(ref) or ref <= 0:
        return None, None, None
    st = 12.0 * np.log2(np.where(ok, f0, np.nan) / ref)
    times = librosa.times_like(f0, sr=sr, hop_length=256)
    return st, ok, times


def melodic_activity(st, ok, times):
    """|d(semitones)/dt| between consecutive voiced frames, octave slips clipped."""
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return None, None
    d_st = np.diff(st[idx])
    d_t = np.diff(times[idx])
    good = d_t > 0
    if not np.any(good):
        return None, None
    d_st, d_t = d_st[good], d_t[good]
    # A jump larger than a tritone in one hop is the tracker, not the singer.
    d_st = np.clip(d_st, -MAX_SEMITONE_JUMP, MAX_SEMITONE_JUMP)
    # activity is attributed to the midpoint of each frame pair
    mid = times[idx][:-1][good] + d_t / 2.0
    return np.abs(d_st), mid


def mass_fraction(activity, mid, lo, hi):
    """Share of total melodic activity falling in [lo, hi]."""
    total = activity.sum()
    if total <= 0:
        return None
    inside = activity[(mid >= lo) & (mid < hi)].sum()
    return float(inside / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--reader", default="R6", help="gold-boundary reader")
    ap.add_argument("--min-per-taam", type=int, default=12)
    args = ap.parse_args()

    if not WINDOWS.exists():
        raise SystemExit(f"missing {WINDOWS.name} — run ctc_align.py --all --write first")
    with open(WINDOWS, encoding="utf-8") as f:
        windows = json.load(f)
    with open(DATASET, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["reader"] == args.reader]

    # Only CTC-derived windows: a proportional fallback would put the syllable
    # where this analysis is trying to find out whether it belongs.
    usable = []
    for r in records:
        w = windows.get(f"{r['reader']}|{r['parasha']}|{r['start']}")
        if w and w["method"] == "ctc" and r["end"] - r["start"] > 0.25:
            usable.append((r, w))
    print(f"{args.reader}: {len(usable):,} words with a CTC stressed window")
    usable = usable[: args.limit]

    by_file = defaultdict(list)
    for r, w in usable:
        by_file[r["audio_file"]].append((r, w))

    real = defaultdict(list)
    null = defaultdict(list)
    rng = np.random.default_rng(0)

    for n, (audio_file, items) in enumerate(by_file.items(), 1):
        path = PROJECT_DIR / audio_file
        if not path.exists():
            continue
        y_full, _ = librosa.load(str(path), sr=SR, mono=True)
        print(f"[{n}/{len(by_file)}] {Path(audio_file).name} ({len(items)} words)",
              flush=True)
        for r, w in items:
            s, e = r["start"], r["end"]
            y = y_full[int(s * SR):int(e * SR)]
            if len(y) < SR * 0.15:
                continue
            st, ok, times = semitone_track(y)
            if st is None:
                continue
            activity, mid = melodic_activity(st, ok, times)
            if activity is None or activity.sum() <= 0:
                continue

            lo, hi = w["start"] - s, w["end"] - s          # syllable, word-relative
            frac = mass_fraction(activity, mid, lo, hi)
            if frac is None:
                continue
            width = hi - lo
            dur = e - s
            real[r["taam"]].append((frac, width / dur))

            # Null: a window of the same width placed at random in the word.
            if width < dur:
                rlo = rng.uniform(0, dur - width)
                fn = mass_fraction(activity, mid, rlo, rlo + width)
                if fn is not None:
                    null[r["taam"]].append(fn)

    print(f"\n{'ta_am':<18}{'n':>6}{'melody in syl':>15}{'syl width':>11}"
          f"{'null':>8}{'lift':>8}  class")
    print("-" * 78)
    rows = []
    for taam, vals in real.items():
        if len(vals) < args.min_per_taam:
            continue
        frac = np.mean([v[0] for v in vals])
        width = np.mean([v[1] for v in vals])
        nl = np.mean(null[taam]) if null.get(taam) else float("nan")
        rows.append((taam, len(vals), frac, width, nl, frac - nl))
    for taam, n, frac, width, nl, lift in sorted(rows, key=lambda x: -x[2]):
        kind = "disjunct" if taam in DISJUNCTIVE else "conjunct"
        print(f"{taam:<18}{n:>6}{frac:>14.1%}{width:>11.1%}{nl:>8.1%}"
              f"{lift:>+8.1%}  {kind}")
    print("-" * 78)
    print("melody in syl = share of the word's melodic activity inside the stressed\n"
          "syllable. syl width = that syllable's share of the word's duration. null =\n"
          "the same-width window placed at random. lift = real minus null: if it is\n"
          "not clearly positive, the stressed syllable is not where the melody lives\n"
          "and per-ta'am windowing has no basis.")

    for kind, want in (("disjunct", True), ("conjunct", False)):
        sel = [r for r in rows if (r[0] in DISJUNCTIVE) == want]
        if sel:
            print(f"\n{kind}: melody in syllable "
                  f"{np.mean([r[2] for r in sel]):.1%} over {len(sel)} ta'amim")


if __name__ == "__main__":
    main()
