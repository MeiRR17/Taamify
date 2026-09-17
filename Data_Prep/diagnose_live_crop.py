"""
Why does the live path contradict 1 in 7 correctly-read words?
==============================================================
Calibration on stored training windows says the verify head contradicts a
correct reading 1.6% of the time. Pushing Ben13's OWN audio -- read
correctly, by definition -- through /api/check_phrase contradicts 14.2% of
words. The head is the same in both cases, so the loss is in how the live
path turns audio into a window.

There are two candidate steps, and they are separable:

  A  stored training window      (features_v3_word/BEN13R_X.npy)
  B  window recomputed from the original audio at the STORED boundaries
     -> isolates the feature-extraction path, with alignment held perfect
  C  window from the live aligner's boundaries
     -> adds alignment error on top

A vs B measures extraction mismatch. B vs C measures alignment error.

The specific suspicion this was written to test: extract_features_v3.py
builds every training window at SR=22050, while webapp/server.py now builds
live windows at SR=16000. Both then keep 87 frames at librosa's default hop
of 512 -- but 87 frames is 2.02 s at 22050 and 2.78 s at 16000, so the same
word lands on a different number of frames than anything the model trained
on. B is therefore computed at BOTH rates, and the one that matches the
stored window is the one the server should be using.

Usage:
  python Data_Prep/diagnose_live_crop.py --words 300
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEAT = PROJECT_DIR / "features_v3_word"
READING = PROJECT_DIR / "data" / "ben13_reading"
OUT = PROJECT_DIR / "results" / "live_crop"


def load_server():
    spec = importlib.util.spec_from_file_location("srv", PROJECT_DIR / "webapp" / "server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


def mel_window(y, start, end, sr, n_mels=128, fmax=8000, max_frames=87,
               pad_sec=0.05, pad_db=-80.0):
    """extract_spectrogram() from Data_Prep/extract_features_v3.py, with the
    sample rate left as a parameter so the two candidates can be compared."""
    import librosa
    s = max(0, int((start - pad_sec) * sr))
    e = min(len(y), int((end + pad_sec) * sr))
    if e <= s or (e - s) < sr * 0.08:
        return None
    mel = librosa.feature.melspectrogram(y=y[s:e], sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_frames - mel_db.shape[1])),
                        mode="constant", constant_values=pad_db)
    return mel_db[:, :max_frames].astype(np.float32)


_ALIGN_CACHE = {}


def live_spans(srv, path, key, by_ref_words, audio_cache):
    """Run the live aligner over one Ben13 block, cached per block, and return
    {word_raw: (start, end)} -- the boundaries /api/check_phrase would use."""
    if key in _ALIGN_CACHE:
        return _ALIGN_CACHE[key]
    words = by_ref_words.get(key)
    if not words:
        _ALIGN_CACHE[key] = None
        return None
    va = srv._load_module("va", "Data_Prep/verse_align.py") if hasattr(srv, "_load_module") else None
    if va is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("va", PROJECT_DIR / "Data_Prep" / "verse_align.py")
        va = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(va)
        spec2 = importlib.util.spec_from_file_location("cv", PROJECT_DIR / "Data_Prep" / "clean_align_v2.py")
        cv = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(cv)
        live_spans.cv = cv
    cv = live_spans.cv
    y16 = audio_cache.get((str(path), 16000))
    if y16 is None:
        return None
    prepared = [{"raw": w, "norm": cv.normalize_word(w)} for w in words]
    try:
        spans, _score, _ = va.align_verse(y16, prepared)
    except Exception:
        spans = None
    out = None
    if spans:
        out = {words[i]: spans[i] for i in spans if i < len(words)}
    _ALIGN_CACHE[key] = out
    return out


def main():
    import librosa

    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=300)
    args = ap.parse_args()

    srv = load_server()
    cal = srv.get_calibration()
    thresh = cal["say_wrong_below"]

    meta = json.loads((FEAT / "BEN13R_meta.json").read_text(encoding="utf-8"))
    X = np.load(FEAT / "BEN13R_X.npy", mmap_mode="r")
    classes = list(srv.get_model()["class_names"])

    rows = json.loads((READING / "maintext.json").read_text(encoding="utf-8"))
    tgspec = importlib.util.spec_from_file_location("tg", PROJECT_DIR / "Data_Prep" / "taam_grammar.py")
    tg = importlib.util.module_from_spec(tgspec); tgspec.loader.exec_module(tg)
    by_ref, by_ref_words = {}, {}
    for r in rows:
        if str(r.get("block")) in ("-1", "0") or not r.get("url"):
            continue
        key = (str(r["parasha"]), str(r["perek"]), str(r["pasuk"]), str(r["block"]))
        by_ref[key] = r["url"]
        by_ref_words[key] = [w for w, _t in tg.verse_tokens(r["text"])]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(meta))

    stats = {k: {"n": 0, "contradicted": 0, "probs": []}
             for k in ("A_stored", "B_22050", "B_16000", "C_live")}
    mse = {"B_22050": [], "B_16000": []}
    audio_cache = {}
    n = 0

    for i in order:
        m = meta[int(i)]
        if m["taam"] not in classes or m.get("align") != "ben13_segment_ctc":
            continue
        perek, pasuk = m["verse_ref"].split(".")[1:3] if m["verse_ref"].count(".") == 2 else (None, None)
        if perek is None:
            continue
        key = (str(m["parasha"]), str(perek), str(pasuk), str(m.get("segment_block")))
        url = by_ref.get(key)
        if not url:
            continue
        path = READING / "yerushalmi" / url
        if not path.exists():
            cands = list((READING / "yerushalmi").glob(f"*/{url.split('/', 1)[1]}")) if "/" in url else []
            if not cands:
                continue
            path = cands[0]

        # A -- the stored training window, exactly as the model saw it
        spec_a = np.array(X[int(i)])
        _, pa = srv.verify_match(spec_a, m["taam"])

        for sr_key, sr in (("B_22050", 22050), ("B_16000", 16000)):
            ck = (str(path), sr)
            if ck not in audio_cache:
                if len(audio_cache) > 24:
                    audio_cache.clear()
                audio_cache[ck] = librosa.load(str(path), sr=sr, mono=True)[0]
            y = audio_cache[ck]
            spec_b = mel_window(y, m["start"], m["end"], sr)
            if spec_b is None:
                continue
            _, pb = srv.verify_match(spec_b, m["taam"])
            stats[sr_key]["n"] += 1
            stats[sr_key]["contradicted"] += 1 if pb < thresh else 0
            stats[sr_key]["probs"].append(pb)
            mse[sr_key].append(float(np.mean((spec_a - spec_b) ** 2)))

        stats["A_stored"]["n"] += 1
        stats["A_stored"]["contradicted"] += 1 if pa < thresh else 0
        stats["A_stored"]["probs"].append(pa)

        # C -- the live aligner's own boundaries for this word, found the way
        # /api/check_phrase finds them. Everything downstream is identical to
        # B_16000, so C minus B is alignment error and nothing else.
        live = live_spans(srv, path, key, by_ref_words, audio_cache)
        if live is not None:
            st_en = live.get(m["word_raw"])
            if st_en:
                y16 = audio_cache[(str(path), 16000)]
                spec_c = mel_window(y16, st_en[0], st_en[1], 16000)
                if spec_c is not None:
                    _, pc = srv.verify_match(spec_c, m["taam"])
                    stats["C_live"]["n"] += 1
                    stats["C_live"]["contradicted"] += 1 if pc < thresh else 0
                    stats["C_live"]["probs"].append(pc)

        n += 1
        if n % 50 == 0:
            print(f"  {n}/{args.words}", flush=True)
        if n >= args.words:
            break

    L = []
    def say(s=""):
        L.append(s); print(s)

    say("=" * 72)
    say("  Where the live path loses correctly-read words")
    say("=" * 72)
    say(f"\n  {n} words from Ben13's own reading, every one read correctly.")
    say(f"  verify threshold in force: {thresh:.3f}\n")
    say(f"  {'window':>12}{'n':>7}{'mean prob':>12}{'CONTRADICTED':>15}"
        f"{'MSE vs stored':>16}")
    say("  " + "-" * 62)
    for k in ("A_stored", "B_22050", "B_16000", "C_live"):
        s = stats[k]
        if not s["n"]:
            continue
        rate = s["contradicted"] / s["n"]
        m_ = f"{np.mean(mse[k]):.2f}" if k in mse and mse[k] else "--"
        say(f"  {k:>12}{s['n']:>7}{np.mean(s['probs']):>12.3f}"
            f"{rate*100:>14.1f}%{m_:>16}")
    say("\n  A_stored  = the training window itself (upper bound)")
    say("  B_22050   = recomputed from audio at the extractor's own rate")
    say("  B_16000   = recomputed at the rate webapp/server.py currently uses")
    say("  C_live    = the live aligner's own boundaries (what the app gets)")
    say("\n  Boundaries are identical in all three, so any gap between the two")
    say("  B rows is caused by sample rate alone.")
    say("\n" + "=" * 72)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "live_crop.json").write_text(json.dumps(
        {k: {"n": v["n"], "contradicted": v["contradicted"],
             "mean_prob": float(np.mean(v["probs"])) if v["probs"] else None}
         for k, v in stats.items()}, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT}/live_crop.json")


if __name__ == "__main__":
    main()
