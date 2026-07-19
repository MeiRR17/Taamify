"""
Feature Extraction v2 — aligned_dataset_v2.json -> consolidated per-reader arrays
=================================================================================
Replaces 2_extract_features.py / 4_extract_features_r2r3.py.

Changes vs v1:
- Input is the verified v2 alignment (aligned_dataset_v2.json).
- One consolidated .npy per reader instead of thousands of tiny files
  (Google Drive chokes on many small files).
- Padding uses -80 dB (true silence floor) instead of 0 dB (v1 accidentally
  padded at MAX loudness, since 0 dB is the ceiling after power_to_db(ref=max)).
- Word placed at window start, right-padded — the same convention inference
  must use (app.py is updated to match in the production phase).

Output (features_v2/):
  <reader>_X.npy      float32 (N, 128, 87)
  <reader>_meta.json  list of N records {taam, word_clean, parasha, verse_ref, start, end}

Usage:
  python Data_Prep/extract_features_v2.py                  # all readers found in the json
  python Data_Prep/extract_features_v2.py --readers R1
"""

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent
ALIGNED_JSON = PROJECT_DIR / "aligned_dataset_v2.json"
OUT_DIR = PROJECT_DIR / "features_v2"
# Per-recording checkpoints live on the local disk, NOT in the Drive-synced
# tree: a crashed/killed run resumes instead of redownloading everything.
PARTS_DIR = Path.home() / ".cache" / "taamify" / "parts"

SR = 22050
N_MELS = 128
FMAX = 8000
MAX_FRAMES = 87          # ~2.02 s at hop 512
PAD_SEC = 0.05           # context around the word, as in v1
PAD_DB = -80.0           # silence floor after power_to_db(ref=np.max)


def extract_word_spectrogram(y_full: np.ndarray, sr: int, start: float, end: float):
    """Mel-spectrogram (128 x 87 dB) for one word slice; None if unusable."""
    s = max(0, int((start - PAD_SEC) * sr))
    e = min(len(y_full), int((end + PAD_SEC) * sr))
    if e <= s or (e - s) < sr * 0.1:
        return None
    y_slice = y_full[s:e]

    mel = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_FRAMES:
        pad = MAX_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=PAD_DB)
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    return mel_db.astype(np.float32)


def wait_for_disk(min_free_gb: float = 1.5, tries: int = 10, wait_s: int = 30):
    """Google Drive evicts its cache asynchronously; give it time under pressure."""
    for _ in range(tries):
        if shutil.disk_usage("/").free / 1e9 >= min_free_gb:
            return True
        print(f"  low disk (<{min_free_gb} GB free), waiting {wait_s}s for Drive eviction...")
        time.sleep(wait_s)
    return False


def load_audio(wav_path: Path, attempts: int = 3):
    """Read + resample, retrying transient Drive/File-Provider failures."""
    for attempt in range(1, attempts + 1):
        try:
            info = sf.info(str(wav_path))
            y, _ = sf.read(str(wav_path), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if info.samplerate != SR:
                y = librosa.resample(y, orig_sr=info.samplerate, target_sr=SR)
            return y
        except Exception as e:
            if attempt == attempts:
                print(f"  FAILED after {attempts} attempts: {wav_path.name}: {e}")
                return None
            wait = 30 * attempt
            print(f"  read error ({e}); retry {attempt}/{attempts - 1} in {wait}s")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=None,
                    help="subset of readers (default: all present in aligned_dataset_v2.json)")
    args = ap.parse_args()

    with open(ALIGNED_JSON, encoding="utf-8") as f:
        records = json.load(f)

    by_reader = defaultdict(lambda: defaultdict(list))  # reader -> audio_file -> records
    for r in records:
        if args.readers and r["reader"] not in args.readers:
            continue
        by_reader[r["reader"]][r["audio_file"]].append(r)

    OUT_DIR.mkdir(exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    for reader in sorted(by_reader):
        specs, metas = [], []
        skipped = Counter()
        files = by_reader[reader]
        print(f"\n{'='*60}\n{reader}: {sum(len(v) for v in files.values())} words "
              f"in {len(files)} recordings\n{'='*60}")

        for audio_file, items in sorted(files.items()):
            part_path = PARTS_DIR / f"{reader}__{Path(audio_file).stem}.npz"
            if part_path.exists():
                part = np.load(part_path, allow_pickle=False)
                part_meta = json.loads(str(part["meta"]))
                specs.extend(part["X"])
                metas.extend(part_meta)
                print(f"  {Path(audio_file).stem}: {len(part_meta)} words (checkpoint)")
                continue

            wav_path = PROJECT_DIR / audio_file
            if not wav_path.exists():
                print(f"  MISSING {audio_file} — skipping {len(items)} words")
                skipped["missing_wav"] += len(items)
                continue
            wait_for_disk()
            t0 = time.time()
            y_full = load_audio(wav_path)
            if y_full is None:
                skipped["unreadable_wav"] += len(items)
                continue
            part_specs, part_metas = [], []
            for item in items:
                spec = extract_word_spectrogram(y_full, SR, item["start"], item["end"])
                if spec is None:
                    skipped["bad_slice"] += 1
                    continue
                part_specs.append(spec)
                part_metas.append({
                    "taam": item["taam"],
                    "word_clean": item["word_clean"],
                    "parasha": item["parasha"],
                    "verse_ref": item["verse_ref"],
                    "start": item["start"],
                    "end": item["end"],
                })
            del y_full
            if part_specs:
                np.savez_compressed(part_path, X=np.stack(part_specs),
                                    meta=json.dumps(part_metas, ensure_ascii=False))
                specs.extend(part_specs)
                metas.extend(part_metas)
            print(f"  {Path(audio_file).stem}: {len(part_specs)}/{len(items)} words "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if not specs:
            print(f"  {reader}: nothing extracted")
            continue

        X = np.stack(specs)
        np.save(OUT_DIR / f"{reader}_X.npy", X)
        with open(OUT_DIR / f"{reader}_meta.json", "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False)

        counts = Counter(m["taam"] for m in metas)
        print(f"\n  {reader}: X {X.shape} ({X.nbytes/1e6:.0f} MB), skipped {dict(skipped)}")
        for taam, n in counts.most_common():
            print(f"    {taam:18s} {n}")


if __name__ == "__main__":
    main()
