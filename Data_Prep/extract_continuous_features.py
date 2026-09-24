"""
Features from the reconstructed continuous reading.
====================================================
Same extractor the corpus used (proven identical, |delta| = 0.00 against the
server's own word_spectrogram), applied to the reconstructed verses at their
KNOWN boundaries. No aligner in the loop, so a model trained here is learning
the tempo, not learning to tolerate alignment error.

The output is drop-in for 00_validate_and_prepare_data.py: BEN13C_X.npy plus
a metadata list carrying parasha, so the canonical parasha holdout applies
unchanged.

Usage:
  python Data_Prep/extract_continuous_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
SRC = P / "data" / "ben13_continuous"
OUT = P / "features_v3_word"
SR = 16000


def main():
    import importlib.util
    import librosa
    spec = importlib.util.spec_from_file_location("srv", P / "webapp" / "server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)

    idx = json.loads((SRC / "index.json").read_text(encoding="utf-8"))
    verses = idx["verses"]
    total = sum(len(v["words"]) for v in verses)
    print(f"  {len(verses):,} פסוקים · {total:,} מילים")

    X = np.lib.format.open_memmap(OUT / "BEN13C_X.npy", mode="w+",
                                  dtype=np.float32, shape=(total, 128, 87))
    meta, n = [], 0
    for vi, v in enumerate(verses):
        y, _ = librosa.load(str(SRC / v["file"]), sr=SR)
        dur = len(y) / SR
        for k, ((a, b), w, t) in enumerate(zip(v["bounds"], v["words"],
                                               v["taamim"])):
            # the same geometry the server serves, anchored on the TRUE onset
            lo = max(0.0, a - srv.CORPUS_LEAD_IN_S)
            sp = srv.word_spectrogram(y, lo, min(dur, lo + srv.CORPUS_WINDOW_S))
            if sp is None:
                continue
            X[n] = sp
            meta.append({"taam": t, "word_raw": w, "word_clean": w,
                         "parasha": v["parasha"], "verse_ref": v["ref"],
                         "word_index": k, "start": a, "end": b,
                         "window": "continuous", "align": "constructed",
                         "low_confidence": False})
            n += 1
        if (vi + 1) % 200 == 0:
            print(f"    {vi+1}/{len(verses)}", flush=True)
    X.flush()
    if n < total:
        np.save(OUT / "BEN13C_X.npy", np.asarray(X[:n]))
    (OUT / "BEN13C_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {n:,} שורות · {OUT/'BEN13C_X.npy'}")


if __name__ == "__main__":
    main()
