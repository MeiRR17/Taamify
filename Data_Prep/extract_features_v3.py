"""
Mel-spectrograms on either the whole word or the stressed syllable.
===================================================================
v2 always cut the whole word. A ta'am is sung on the stressed syllable, and the
alignment measured that syllable at ~45% of the word window — so the v2 window
spends more than half its frames on audio that carries no ta'am at all. This
version cuts either window from the same records, which is what makes the two
directly comparable: same words, same labels, same split, one variable.

  --window word       reproduces v2
  --window stressed   the syllable window cached by ctc_align.py

Optional `--speaker-norm` subtracts each *recording's* mean/variance from the
spectrogram. A reader's voice is a constant offset in mel space; removing it is
the cheapest attack on the gap between the random split and LORO, where the
network otherwise has a channel that identifies the reader instead of the mark.

Both windows are padded/truncated to the same 128 x 87 shape as v2, so the
existing architectures in 10_train_pipeline.py load them unchanged.

Usage:
  python Data_Prep/extract_features_v3.py --window stressed
  python Data_Prep/extract_features_v3.py --window word --speaker-norm
"""

import argparse
import os
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET = PROJECT_DIR / "aligned_dataset_merged.json"
WINDOWS = PROJECT_DIR / "stressed_windows.json"
# R1/R2's word boundaries came from Whisper, which sits a median 200 ms early at
# the start and 284 ms early at the end (measured against R6's gold onsets), so
# its windows open on the previous word and close mid-syllable. verse_align.py
# re-derives them from the audio; where it produced a boundary, it wins over the
# stored one. R6 keeps its human-verified onsets.
VERSE_WINDOWS = PROJECT_DIR / "verse_aligned_windows.json"

# Must match the extractor that produced the corpus a model is trained
# against. Ben13's words are built at ctc_align.SR = 16000, and mixing the
# two in one training pool silently compares different time scales: with
# librosa's default hop a frame is 23.2ms at 22050 and 32ms at 16000, so the
# same 87 frames span 2.02s versus 2.78s. That mismatch made a cross-voice
# test read 4.38% -- chance -- and it is the same bug that broke the live
# server. Overridable so the old 22050 features can still be reproduced.
SR = int(os.environ.get("TAAMIFY_SR", "16000"))
N_MELS = 128
FMAX = 8000
MAX_FRAMES = 87          # ~2.02 s at hop 512, as in v2
PAD_SEC = 0.05
PAD_DB = -80.0
MIN_SLICE_SEC = 0.08


def extract_spectrogram(y_full, start, end):
    """Mel-spectrogram (128 x 87 dB) for one slice; None if unusable."""
    s = max(0, int((start - PAD_SEC) * SR))
    e = min(len(y_full), int((end + PAD_SEC) * SR))
    if e <= s or (e - s) < SR * MIN_SLICE_SEC:
        return None
    mel = librosa.feature.melspectrogram(y=y_full[s:e], sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])),
                        mode="constant", constant_values=PAD_DB)
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    return mel_db.astype(np.float32)


def load_audio(path: Path):
    try:
        info = sf.info(str(path))
        y, _ = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if info.samplerate != SR:
            y = librosa.resample(y, orig_sr=info.samplerate, target_sr=SR)
        return y
    except Exception:
        try:                       # soundfile has no mp3 on every build
            y, _ = librosa.load(str(path), sr=SR, mono=True)
            return y
        except Exception as e:
            print(f"    unreadable: {path.name}: {e}")
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["word", "stressed"], default="stressed")
    ap.add_argument("--speaker-norm", action="store_true",
                    help="per-recording mean/variance normalization")
    ap.add_argument("--readers", nargs="+", default=None)
    ap.add_argument("--no-verse-align", action="store_true",
                    help="ignore verse_aligned_windows.json (ablation)")
    ap.add_argument("--jitter-ms", type=float, default=0.0,
                    help="Gaussian timing noise (sd, milliseconds) added independently "
                         "to each window's start and end. For the noise-control "
                         "experiment: R7's alignment is measurably less confident than "
                         "R1/R2/R6's (median CTC score -2.79 vs -1.91), and noisy "
                         "boundaries attenuate any real effect toward zero. Degrading a "
                         "reader with a known-strong effect by a known amount says how "
                         "much of R7's collapse noise alone could explain.")
    ap.add_argument("--jitter-readers", nargs="+", default=None,
                    help="restrict --jitter-ms to these readers (default: all)")
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default features_v3_<window>)")
    args = ap.parse_args()

    out_dir = PROJECT_DIR / (args.out or f"features_v3_{args.window}"
                             + ("_spknorm" if args.speaker_norm else ""))
    out_dir.mkdir(exist_ok=True)
    parts_dir = Path.home() / ".cache" / "taamify" / f"parts_v3_{out_dir.name}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    jitter_sd = args.jitter_ms / 1000.0
    rng = np.random.default_rng(args.jitter_seed)
    if jitter_sd > 0:
        who = args.jitter_readers or "all readers"
        print(f"timing jitter: sd {args.jitter_ms:.0f} ms on {who}")

    with open(DATASET, encoding="utf-8") as f:
        records = json.load(f)

    windows = {}
    if args.window == "stressed":
        if not WINDOWS.exists():
            raise SystemExit(f"missing {WINDOWS.name} — run ctc_align.py --all --write first")
        with open(WINDOWS, encoding="utf-8") as f:
            windows = json.load(f)
        print(f"stressed windows available: {len(windows):,}")

    verse_windows = {}
    if VERSE_WINDOWS.exists() and not args.no_verse_align:
        with open(VERSE_WINDOWS, encoding="utf-8") as f:
            verse_windows = json.load(f)
        print(f"acoustic word boundaries available: {len(verse_windows):,}")
    elif args.window == "word":
        print("WARNING: no verse-aligned boundaries — R1/R2 will use Whisper's, "
              "which are known to be off by ~200 ms")

    by_reader = defaultdict(lambda: defaultdict(list))
    for r in records:
        if args.readers and r["reader"] not in args.readers:
            continue
        by_reader[r["reader"]][r["audio_file"]].append(r)

    for reader in sorted(by_reader):
        specs, metas = [], []
        skipped = Counter()
        files = by_reader[reader]
        total = sum(len(v) for v in files.values())
        print(f"\n{'=' * 60}\n{reader}: {total:,} words in {len(files)} recordings"
              f"\n{'=' * 60}")

        for audio_file, items in sorted(files.items()):
            part_path = parts_dir / f"{reader}__{Path(audio_file).stem}.npz"
            if part_path.exists():
                part = np.load(part_path, allow_pickle=False)
                specs.extend(part["X"])
                metas.extend(json.loads(str(part["meta"])))
                print(f"  {Path(audio_file).stem}: checkpoint", flush=True)
                continue

            path = PROJECT_DIR / audio_file
            if not path.exists():
                skipped["missing_audio"] += len(items)
                continue
            t0 = time.time()
            y_full = load_audio(path)
            if y_full is None:
                skipped["unreadable"] += len(items)
                continue

            part_specs, part_metas = [], []
            for item in items:
                key = f"{item['reader']}|{item['parasha']}|{item['start']}"
                va = verse_windows.get(key)
                if args.window == "stressed":
                    # A stressed window sitting inside a bad word window is bad
                    # too, so prefer the one derived from the acoustic boundary.
                    if va and "start" in va:
                        start, end, method = va["start"], va["end"], "verse_ctc"
                    else:
                        w = windows.get(key)
                        if w is None:
                            skipped["no_stressed_window"] += 1
                            continue
                        start, end, method = w["start"], w["end"], w["method"]
                elif va:
                    start, end, method = va["word_start"], va["word_end"], "verse_ctc"
                else:
                    start, end, method = item["start"], item["end"], "stored"

                if jitter_sd > 0 and (args.jitter_readers is None
                                      or item["reader"] in args.jitter_readers):
                    # start and end move independently, so duration is perturbed
                    # too — a misaligned window is wrong about where the word
                    # begins *and* how long it lasts, which is what the Whisper
                    # boundary measurements showed.
                    start = start + rng.normal(0.0, jitter_sd)
                    end = end + rng.normal(0.0, jitter_sd)
                    if end - start < MIN_SLICE_SEC:
                        skipped["jitter_collapsed"] += 1
                        continue
                    method = f"{method}+jitter"

                spec = extract_spectrogram(y_full, start, end)
                if spec is None:
                    skipped["bad_slice"] += 1
                    continue
                part_specs.append(spec)
                part_metas.append({
                    "taam": item["taam"],
                    "word_clean": item["word_clean"],
                    "parasha": item["parasha"],
                    "verse_ref": item["verse_ref"],
                    "start": start, "end": end,
                    "window": args.window, "align": method,
                })

            if part_specs:
                X_part = np.stack(part_specs)
                if args.speaker_norm:
                    # Per recording, not per reader: channel and room differ
                    # between a reader's uploads as much as between readers.
                    X_part = (X_part - X_part.mean()) / (X_part.std() + 1e-6)
                np.savez_compressed(part_path, X=X_part,
                                    meta=json.dumps(part_metas, ensure_ascii=False))
                specs.extend(X_part)
                metas.extend(part_metas)
            print(f"  {Path(audio_file).stem}: {len(part_specs)}/{len(items)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del y_full

        if not specs:
            print(f"  {reader}: nothing extracted · skipped {dict(skipped)}")
            continue

        X = np.stack(specs)
        np.save(out_dir / f"{reader}_X.npy", X)
        with open(out_dir / f"{reader}_meta.json", "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False)

        aligns = Counter(m["align"] for m in metas)
        print(f"\n  {reader}: X {X.shape} ({X.nbytes / 1e6:.0f} MB)")
        print(f"  alignment: {dict(aligns)}")
        if skipped:
            print(f"  skipped: {dict(skipped)}")

    print(f"\nwrote -> {out_dir.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
