"""
Turn consented learner recordings into training rows — the growth loop.
========================================================================
The corpus is one voice. Measured against that fact: the shipped model
scores 15/16 on Ben13's own clips and 0/2 on a different reader, at high
confidence. No amount of compute fixes that; only other voices do.

The decision taken was to ship to production and let voices accumulate
through use rather than waiting for a collection campaign. This is the
script that makes that decision pay off. Every consented phrase recorded in
the app arrives already paired with the exact words and ta'amim it was read
against, so it needs no manual labelling: the same CTC aligner that built
the whole corpus (verse_align.align_verse) cuts it into per-word
spectrograms in the identical format.

Each distinct voice_id becomes its own reader (USER01, USER02, ...), which
matters more than the row count: readers are what LORO measures, so a second
real voice immediately makes the honest generalisation number computable for
the first time.

Only rows whose alignment succeeded are kept, and each carries its CTC score
so a later pass can drop the weak ones -- the same provenance discipline as
the rest of the corpus.

Usage:
  python Data_Prep/ingest_collected_phrases.py            # all new clips
  python Data_Prep/ingest_collected_phrases.py --min-score -2.0
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import normalize_word          # noqa: E402
from ctc_align import SR                           # noqa: E402
from verse_align import align_verse                # noqa: E402

PHRASES = P / "app_assets" / "collected_phrases"
LOG = PHRASES / "phrases_log.jsonl"
OUT = P / "features_v3_word"
STATE = PHRASES / "ingested.json"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=-3.0,
                    help="drop phrases whose CTC alignment scored below this")
    args = ap.parse_args()

    if not LOG.exists():
        print("no collected phrases yet — the app writes them as learners practise")
        return
    done = set()
    if STATE.exists():
        done = set(json.loads(STATE.read_text()).get("clip_ids", []))

    rows = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Only what the gatekeeper cleared. Without this, a learner who sang the
    # WRONG ta'am contributes a row asserting that wrong sound IS the ta'am --
    # teaching the model the very mistakes it exists to catch.
    safe_path = PHRASES / "training_safe.json"
    if safe_path.exists():
        allowed = set(json.loads(safe_path.read_text())["allowed_clip_ids"])
        blocked = [r for r in rows if r["clip_id"] not in allowed]
        rows = [r for r in rows if r["clip_id"] in allowed]
        if blocked:
            print(f"gatekeeper blocked {len(blocked)} clip(s) from training "
                  f"(Data_Prep/voice_intake.py)")
    else:
        print("WARNING: no training_safe.json -- run voice_intake.py --promote "
              "first, or every recording trains regardless of whether it was "
              "read correctly")

    fresh = [r for r in rows if r["clip_id"] not in done]
    print(f"{len(rows)} phrases logged, {len(fresh)} not yet ingested")
    if not fresh:
        return

    by_voice = defaultdict(list)
    for r in fresh:
        by_voice[r.get("voice_id") or "anon"].append(r)
    print(f"{len(by_voice)} distinct voice(s): {sorted(by_voice)}")

    # voice -> reader id, stable across runs
    mapping = {}
    if STATE.exists():
        mapping = json.loads(STATE.read_text()).get("voices", {})
    for v in sorted(by_voice):
        if v not in mapping:
            mapping[v] = f"USER{len(mapping) + 1:02d}"

    ingested = set(done)
    for voice, recs in by_voice.items():
        reader = mapping[voice]
        specs, meta, stats = [], [], Counter()
        # keep whatever this reader already has, so runs accumulate
        xp, mp = OUT / f"{reader}_X.npy", OUT / f"{reader}_meta.json"
        if xp.exists() and mp.exists():
            specs = list(np.load(xp))
            meta = json.loads(mp.read_text(encoding="utf-8"))
            print(f"  {reader}: {len(meta)} existing rows")

        for r in recs:
            wav = PHRASES / f"{r['clip_id']}.wav"
            if not wav.exists():
                stats["audio_missing"] += 1
                continue
            try:
                y, _ = librosa.load(str(wav), sr=SR)
            except Exception:
                stats["unreadable"] += 1
                continue
            words = [{"raw": w, "norm": normalize_word(w)} for w in r["words"]]
            try:
                spans, score, _ = align_verse(y, words)
            except Exception:
                spans, score = None, None
            if spans is None:
                stats["align_failed"] += 1
                continue
            if score is not None and score < args.min_score:
                stats["low_score"] += 1
                continue
            kept = 0
            for i, taam in enumerate(r["taamim"]):
                if not taam or i not in spans:
                    continue
                st_, en = spans[i]
                spec = word_spectrogram(y, st_, en)
                if spec is None:
                    continue
                specs.append(spec)
                meta.append({"taam": taam, "word_clean": words[i]["norm"],
                             "word_raw": r["words"][i], "reader": reader,
                             "parasha": "user", "verse_ref": r["clip_id"],
                             "start": round(st_, 3), "end": round(en, 3),
                             "window": "word", "align": "user_phrase_ctc",
                             "ctc_score": round(float(score), 3) if score is not None else None})
                kept += 1
            stats["ok"] += 1
            stats["words"] += kept
            ingested.add(r["clip_id"])

        if specs:
            np.save(xp, np.stack(specs))
            mp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            c = Counter(m["taam"] for m in meta)
            print(f"  {reader}: {len(meta):,} words, {len(c)} classes  {dict(stats)}")
        else:
            print(f"  {reader}: nothing usable  {dict(stats)}")

    STATE.write_text(json.dumps({"clip_ids": sorted(ingested), "voices": mapping},
                                indent=1), encoding="utf-8")
    readers = [mapping[v] for v in sorted(by_voice)]
    print(f"\nnew readers ready: {' '.join(readers)}")
    print("the honest cross-voice number is now computable, e.g.:")
    print(f"  python 00_validate_and_prepare_data.py --readers BEN13R {readers[0]} "
          f"--num-classes 21 --split loro --holdout {readers[0]} --tag voice2")


if __name__ == "__main__":
    main()
