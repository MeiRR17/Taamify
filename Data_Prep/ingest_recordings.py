"""
Ingest recordings exported from the Taamim Studio page.
========================================================
The studio (published artifact) stores each recording in the browser and in
the page's own private store, and exports them as one JSON file. There is no
tool here that can read that store directly, so the export file is the only
bridge: drop it in incoming/ and this turns it into a reader the rest of the
pipeline already understands.

Why this matters more than any training run queued beside it: every voice in
the corpus today -- R1, R2, R8 and Ben13 -- is an adult male ba'al kri'ah,
and Ben13, now the only corpus, is a SINGLE voice. The honest 86.9% comes
from holding out whole parashot, which controls for memorised audio but not
for the voice itself. A second Sephardic-Jerusalem voice is the first real
test of whether the model learned the ta'amim or learned one reader.

What it does per recording:
  1. decode the base64 audio (webm/opus from MediaRecorder) via ffmpeg
  2. align the words of that verse against the audio with the same CTC
     aligner the rest of the corpus used (verse_align.align_verse)
  3. cut each word and store the identical 128x87 mel-dB feature
  4. write features_v3_word/USER_X.npy + _meta.json -- a new reader, so
     every existing script, split and model can consume it unchanged

Usage:
  python Data_Prep/ingest_recordings.py                     # newest file in incoming/
  python Data_Prep/ingest_recordings.py --file path.json --reader USER
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import normalize_word          # noqa: E402
from ctc_align import SR                           # noqa: E402
from verse_align import align_verse                # noqa: E402
import taam_grammar as tg                          # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
INCOMING = PROJECT_DIR / "incoming"
OUT_DIR = PROJECT_DIR / "features_v3_word"
AUDIO_OUT = PROJECT_DIR / "app_assets" / "user_recordings"
N_MELS, FMAX, MAX_FRAMES, PAD_DB = 128, 8000, 87, -80.0


def word_spectrogram(y, start, end):
    s = max(0, int((start - 0.05) * SR))
    e = min(len(y), int((end + 0.05) * SR))
    if e <= s or (e - s) < SR * 0.1:
        return None
    mel = librosa.feature.melspectrogram(y=y[s:e], sr=SR, n_mels=N_MELS, fmax=FMAX)
    db = librosa.power_to_db(mel, ref=np.max)
    if db.shape[1] < MAX_FRAMES:
        db = np.pad(db, ((0, 0), (0, MAX_FRAMES - db.shape[1])),
                    mode="constant", constant_values=PAD_DB)
    return db[:, :MAX_FRAMES].astype(np.float32)


def decode(data_url: str, dst: Path) -> bool:
    """data: URL -> 16 kHz mono wav, via ffmpeg (the browser gives webm/opus)."""
    if "," not in data_url:
        return False
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(raw)
        tmp = f.name
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-ac", "1", "-ar", str(SR), str(dst)],
                       capture_output=True)
    Path(tmp).unlink(missing_ok=True)
    return r.returncode == 0 and dst.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--reader", default="USER")
    args = ap.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        cands = sorted(INCOMING.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print("nothing in incoming/ — waiting for an export from the studio page")
            return
        path = cands[-1]
    print(f"reading {path.name}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    recs = payload.get("recordings", payload if isinstance(payload, list) else [])
    verses = {v["ref"]: v for v in
              json.loads((PROJECT_DIR / "app_assets" / "record_verses.json")
                         .read_text(encoding="utf-8"))}
    print(f"{len(recs)} recordings")

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    specs, meta, stats = [], [], {"ok": 0, "no_verse": 0, "decode_fail": 0,
                                  "align_fail": 0}
    for rec in recs:
        ref = rec.get("ref")
        v = verses.get(ref)
        if not v:
            stats["no_verse"] += 1
            continue
        wav = AUDIO_OUT / (ref.replace(" ", "_").replace(":", "_") + ".wav")
        if not decode(rec.get("audio", ""), wav):
            stats["decode_fail"] += 1
            continue
        y, _ = librosa.load(str(wav), sr=SR)
        tokens = tg.verse_tokens(v["text"])
        words = [{"raw": w, "norm": normalize_word(w)} for w, _ in tokens]
        spans, score, _ = align_verse(y, words)
        if spans is None:
            stats["align_fail"] += 1
            print(f"  {ref}: alignment failed")
            continue
        kept = 0
        for wi, (_, taam) in enumerate(tokens):
            if taam is None or wi not in spans:
                continue
            st, en = spans[wi]
            spec = word_spectrogram(y, st, en)
            if spec is None:
                continue
            specs.append(spec)
            meta.append({"taam": taam, "word_clean": words[wi]["norm"],
                         "verse_ref": ref, "reader": args.reader,
                         "parasha": "user", "start": round(st, 3),
                         "end": round(en, 3), "window": "word",
                         "align": "user_ctc",
                         "ctc_score": round(float(score), 3) if score is not None else None})
            kept += 1
        stats["ok"] += 1
        print(f"  {ref}: {kept} words  (ctc {score:.2f})" if score is not None
              else f"  {ref}: {kept} words")

    print(f"\n{stats}")
    if not specs:
        print("nothing extracted")
        return
    arr = np.stack(specs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{args.reader}_X.npy", arr)
    (OUT_DIR / f"{args.reader}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    c = Counter(m["taam"] for m in meta)
    print(f"{args.reader}: {arr.shape} -> features_v3_word/{args.reader}_X.npy")
    print("classes captured:")
    for k, n in c.most_common():
        print(f"  {k:<18} {n}")
    done = path.parent / "processed"
    done.mkdir(exist_ok=True)
    path.rename(done / path.name)
    print(f"\nmoved {path.name} -> incoming/processed/")


if __name__ == "__main__":
    main()
