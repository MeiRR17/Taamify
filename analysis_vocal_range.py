"""
analysis_vocal_range.py — measure a singer's usable range from a recording
==========================================================================
Before a song gets a key, the voice needs a measurement. The chorus is the
loudest, highest, most repeated part of a song; if it sits above the singer's
comfortable range the take is strained, and if it sits in the middle the
chorus never feels like it lifts. Both are key-choice mistakes, and both are
avoidable by measuring first.

Reuses the pitch-tracking approach of make_reference_contours.py, but where
that file normalizes pitch away (semitones relative to each sample's own
median, so readers become comparable), here absolute pitch IS the signal.

What it measures
----------------
  full range        lowest..highest voiced pitch — what you *can* hit
  comfortable range the sub-range that is loud and pitch-stable — what you
                    can hit *repeatedly, on take 7, in tune*

The second one is the useful one, and it is not a matter of opinion: notes at
the edge of a singer's range wobble (wider f0 spread) and go quiet (lower
RMS) compared to the middle. This script bins every voiced frame by semitone
and reports both quantities per semitone, so the drop-off is visible rather
than asserted.

Recording the input
-------------------
Sing a slow chromatic slide on "ah", from the lowest note you can phonate to
the highest, in one breath group per direction. Two or three passes in one
file is better than one. Any format GarageBand exports (m4a/wav/aiff) works.

Usage
-----
    venv/bin/python analysis_vocal_range.py range.m4a
    venv/bin/python analysis_vocal_range.py range.m4a --minor    # key advice
    venv/bin/python analysis_vocal_range.py phrase.m4a --notes   # transcribe

Caveat: range alone does not determine voice type. Tessitura (where the voice
sits *easily* for a long time) and timbre matter as much, and a single
recording on one day is one sample. Treat the classification as a hint.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

# Male voices essentially never leave this window; a wider search just gives
# pyin more room to produce octave errors on breathy or creaky frames.
FMIN_NOTE = "C2"   # ~65 Hz
FMAX_NOTE = "C6"   # ~1047 Hz

# A semitone needs this many voiced frames before we trust its statistics.
# At the default hop (512 @ 22050 Hz ≈ 23 ms) this is roughly a third of a
# second of singing — below that, a passing glide artifact could define an
# entire semitone's "stability".
MIN_FRAMES_PER_SEMITONE = 12

# Comfort thresholds, applied to each semitone relative to the singer's own
# best semitone. They are deliberately generous: the goal is to find where
# the voice clearly falls off, not to draw a precise border.
COMFORT_MIN_RMS_RATIO = 0.45    # at least 45% of the loudest semitone's RMS
COMFORT_MAX_SPREAD_CENTS = 45.0 # f0 IQR within the semitone


def note_name(midi):
    """MIDI number -> scientific pitch name, e.g. 57 -> A3."""
    return librosa.midi_to_note(int(round(midi)), unicode=False)


def _load_audio(path, sr):
    """librosa.load, with an ffmpeg decode as a fallback for anything
    libsndfile can't read directly.

    This build of librosa calls straight into soundfile with no fallback of
    its own, and soundfile only speaks the formats libsndfile does — wav,
    aiff, flac. A phone's voice recorder produces m4a (AAC), which is
    exactly the file this script exists to accept, so without this fallback
    the first real recording anyone makes would crash the tool. ffmpeg
    already ships with Homebrew and decodes essentially anything; when it's
    missing we fail with a install hint instead of a stack trace.
    """
    try:
        return librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        pass

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            f"Could not read {path} directly, and ffmpeg (needed to decode "
            "it) is not on PATH. Install it with: brew install ffmpeg"
        )
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(sr), tmp.name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"ffmpeg could not decode {path}:\n{result.stderr[-500:]}"
            )
        return librosa.load(tmp.name, sr=sr, mono=True)


def extract_pitch(path, sr=22050):
    """Voiced f0 (in MIDI floats) plus per-frame RMS, aligned frame-for-frame."""
    y, sr = _load_audio(path, sr)
    y, _ = librosa.effects.trim(y, top_db=35)

    f0, voiced, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(FMIN_NOTE),
        fmax=librosa.note_to_hz(FMAX_NOTE),
        sr=sr,
    )
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

    n = min(len(f0), len(rms))
    f0, voiced, voiced_prob, rms = f0[:n], voiced[:n], voiced_prob[:n], rms[:n]

    # Require both the voicing decision and a confident probability. pyin
    # flags a frame voiced on a fairly low bar, and unconfident frames at the
    # extremes are exactly the ones that would inflate the measured range.
    keep = voiced & (voiced_prob > 0.6) & np.isfinite(f0)
    if keep.sum() == 0:
        raise SystemExit(
            "No confident voiced frames found. Is the file mostly silence, "
            "or the singing very breathy? Try recording closer to the mic."
        )
    return librosa.hz_to_midi(f0[keep]), rms[keep], sr, len(y) / sr


def per_semitone_stats(midi, rms):
    """Group frames by semitone; return one row per semitone, low to high."""
    semis = np.round(midi).astype(int)
    rows = []
    for s in range(semis.min(), semis.max() + 1):
        sel = semis == s
        n = int(sel.sum())
        if n < MIN_FRAMES_PER_SEMITONE:
            continue
        cents = (midi[sel] - s) * 100.0
        rows.append({
            "midi": s,
            "n": n,
            "rms": float(np.median(rms[sel])),
            "spread": float(np.percentile(cents, 75) - np.percentile(cents, 25)),
        })
    return rows


def comfortable_span(rows):
    """Longest run of consecutive semitones that pass both comfort tests."""
    if not rows:
        return []
    best_rms = max(r["rms"] for r in rows)
    ok = {
        r["midi"] for r in rows
        if r["rms"] >= best_rms * COMFORT_MIN_RMS_RATIO
        and r["spread"] <= COMFORT_MAX_SPREAD_CENTS
    }
    runs, cur = [], []
    for r in rows:
        if r["midi"] in ok:
            cur.append(r["midi"])
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return max(runs, key=len) if runs else []


def classify(low, high):
    """Rough voice-type hint: the textbook range the measurement best fits."""
    span = high - low
    if span <= 0:
        return "unknown"
    scored = []
    for label, lo_note, hi_note in [
        ("bass",     "E2", "E4"),
        ("baritone", "G2", "G4"),
        ("tenor",    "C3", "C5"),
    ]:
        lo = librosa.note_to_midi(lo_note)
        hi = librosa.note_to_midi(hi_note)
        overlap = max(0, min(high, hi) - max(low, lo))
        scored.append((overlap / span, label))
    best, label = max(scored)
    # Below this the measurement straddles two categories, and naming one
    # would be a coin flip dressed up as a result.
    return label if best > 0.75 else "between the standard categories"


def report(rows, comfy, duration):
    lo_all, hi_all = rows[0]["midi"], rows[-1]["midi"]

    print(f"\n  {duration:.1f}s analysed, {len(rows)} semitones with enough data\n")
    print("  FULL RANGE")
    print(f"    {note_name(lo_all)} .. {note_name(hi_all)}"
          f"   ({hi_all - lo_all} semitones,"
          f" {(hi_all - lo_all) / 12:.1f} octaves)")
    print(f"    voice-type hint: {classify(lo_all, hi_all)}")

    if not comfy:
        print("\n  COMFORTABLE RANGE: could not isolate one — the recording is")
        print("  too uneven in level. Re-record at a steady distance from the mic.")
        return None

    lo_c, hi_c = comfy[0], comfy[-1]
    print("\n  COMFORTABLE RANGE  (loud and pitch-stable)")
    print(f"    {note_name(lo_c)} .. {note_name(hi_c)}"
          f"   ({hi_c - lo_c} semitones)")

    print("\n  PER-SEMITONE EVIDENCE     (* = comfortable)")
    print("    note    frames   level   wobble")
    best_rms = max(r["rms"] for r in rows)
    for r in rows:
        mark = "*" if r["midi"] in comfy else " "
        bar = "#" * int(round(20 * r["rms"] / best_rms))
        print(f"  {mark} {note_name(r['midi']):<6} {r['n']:>6}"
              f"   {bar:<20} {r['spread']:>5.0f}c")
    return lo_c, hi_c


def key_advice(lo_c, hi_c, minor):
    """Suggest tonics that put the chorus peak near the top of comfort."""
    # Leave two semitones of headroom: the top of the comfortable range is
    # where the voice is still stable, not where it should live all night.
    peak = hi_c - 2
    quality = "minor" if minor else "major"

    # In pop/rock choruses the melody most often peaks on the 5th degree or
    # on the octave root. Solve backwards: what tonic puts one of those at
    # the peak note, while keeping the tonic itself inside comfortable range?
    print(f"\n  KEY SUGGESTIONS  ({quality}, chorus peaking near"
          f" {note_name(peak)})")
    print("    peak degree   key           tonic sits at")
    found = False
    for degree_name, interval in [("5th", 7), ("octave root", 12)]:
        tonic = peak - interval
        while tonic < lo_c:
            tonic += 12
        if lo_c <= tonic <= hi_c:
            found = True
            root = note_name(tonic).rstrip("0123456789")
            print(f"    {degree_name:<13} {root} {quality:<10}"
                  f"  {note_name(tonic)}")
    if not found:
        print("    comfortable range too narrow for a standard chorus shape —")
        print("    consider a chorus that peaks on the 3rd instead")
    print("\n  Sanity check: sing the chorus's highest note in each candidate")
    print("  key ten times. The one you can still hit on the tenth is the key.")


def transcribe(midi):
    """Print the note sequence of a sung phrase, merging repeated notes."""
    semis = np.round(midi).astype(int)
    out, prev, run = [], None, 0
    for s in semis:
        if s == prev:
            run += 1
        else:
            if prev is not None and run >= MIN_FRAMES_PER_SEMITONE // 2:
                out.append(note_name(prev))
            prev, run = s, 1
    if prev is not None and run >= MIN_FRAMES_PER_SEMITONE // 2:
        out.append(note_name(prev))
    print("\n  NOTES: " + " ".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="recording to analyse")
    ap.add_argument("--minor", action="store_true",
                    help="suggest minor keys (default: major)")
    ap.add_argument("--notes", action="store_true",
                    help="transcribe the sung notes instead of measuring range")
    args = ap.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"No such file: {args.audio}")

    midi, rms, _sr, duration = extract_pitch(args.audio)

    if args.notes:
        transcribe(midi)
        return

    rows = per_semitone_stats(midi, rms)
    if not rows:
        raise SystemExit(
            f"Every semitone had fewer than {MIN_FRAMES_PER_SEMITONE} frames. "
            "Sing the slide more slowly."
        )
    comfy = comfortable_span(rows)
    span = report(rows, comfy, duration)
    if span:
        key_advice(span[0], span[1], args.minor)


if __name__ == "__main__":
    main()
