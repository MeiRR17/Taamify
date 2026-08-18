"""
Character-level forced alignment inside a word, and the stressed-syllable window.
=================================================================================
`stressed_syllable.py` says *which letters* carry the ta'am. This module says
*when* those letters are sung.

MFA ships no Hebrew acoustic model (67 languages, Hebrew is not among them), and
training one would need a pronunciation dictionary we would have to write. A
character-level CTC model gets us the same thing without any of that:
`imvladikon/wav2vec2-xls-r-300m-hebrew` has a 34-symbol vocabulary that is
exactly the Hebrew alphabet plus a word delimiter, so the consonantal spelling
already *is* the target sequence.

Alignment runs inside one word at a time, over the word window we already have
(gold onsets for R6, Whisper for R1/R2) plus padding. Short windows with known
text make the alignment both cheap and hard to get wrong — a whole-recording
alignment would be neither.

Frames are the wav2vec2 stride: 320 samples at 16 kHz = 20 ms.

Usage:
  python Data_Prep/ctc_align.py --demo                        # a few words, printed
  python Data_Prep/ctc_align.py --dataset aligned_dataset_pt.json --limit 500
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stressed_syllable import stressed_syllable  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
# huggingface_hub's xet transfer backend dies behind an inspecting proxy, and
# from_pretrained then silently re-downloads 1.2 GB on every call. Fetch the
# repo once with curl into models/ and load from disk.
LOCAL_MODEL = PROJECT_DIR / "models" / "wav2vec2-he"
MODEL_ID = str(LOCAL_MODEL) if (LOCAL_MODEL / "model.safetensors").exists() \
    else "imvladikon/wav2vec2-xls-r-300m-hebrew"
SR = 16000
PAD_SEC = 0.15          # context around the word window before aligning

_model = _vocab = _blank = None


def load_model(device: str = None):
    """Lazily load the CTC model; MPS on Apple Silicon, else CPU."""
    global _model, _vocab, _blank
    if _model is not None:
        return _model, _vocab, _blank
    try:
        import truststore
        truststore.inject_into_ssl()      # TLS through inspecting proxies
    except ImportError:
        pass
    from transformers import AutoModelForCTC, AutoProcessor

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(device).eval()
    vocab = proc.tokenizer.get_vocab()
    _model, _vocab, _blank = (model, device), vocab, vocab["[PAD]"]
    return _model, _vocab, _blank


_audio_cache = {}


def load_recording(path: Path) -> np.ndarray:
    """Whole recording at 16 kHz mono, cached.

    Slicing in memory rather than re-opening per word matters: librosa's
    `offset` still decodes an mp3 from the top, so per-word loading costs
    O(words x recording length).
    """
    key = str(path)
    if key not in _audio_cache:
        import librosa
        _audio_cache.clear()          # one recording at a time; these are large
        y, _ = librosa.load(key, sr=SR, mono=True)
        _audio_cache[key] = y
    return _audio_cache[key]


def slice_window(y_full: np.ndarray, start: float, end: float):
    """(slice, slice_offset_seconds) for a padded word window."""
    offset = max(0.0, start - PAD_SEC)
    s = int(offset * SR)
    e = min(len(y_full), int((end + PAD_SEC) * SR))
    return y_full[s:e], offset


def align_word(y: np.ndarray, letters: str):
    """Per-letter onset frames for `letters` within the audio `y`.

    Returns (onsets, n_frames, frame_sec), where onsets[i] is the frame at which
    letter i is emitted, or (None, None, None) if the word cannot be aligned.

    CTC is peaky: it spends almost every frame on blank and fires each character
    on one or two frames. The emitted run is therefore *not* the duration of the
    sound — a letter's audio runs from its own onset to the next letter's onset.
    Callers must build spans that way; measuring the run itself yields the
    20 ms spikes rather than syllables.
    """
    (model, device), vocab, blank = load_model()
    ids = [vocab[c] for c in letters if c in vocab]
    if not ids or len(y) < SR * 0.05:
        return None, None, None

    with torch.inference_mode():
        logits = model(torch.from_numpy(y).unsqueeze(0).to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1).cpu()

    n_frames = log_probs.shape[1]
    if n_frames < len(ids):          # window too short to host every letter
        return None, None, None

    import torchaudio.functional as AF
    targets = torch.tensor([ids], dtype=torch.int32)
    try:
        aligned, scores = AF.forced_align(log_probs.float(), targets, blank=blank)
        spans = AF.merge_tokens(aligned[0], scores[0], blank=blank)
    except (RuntimeError, ValueError):
        return None, None, None

    if len(spans) != len(ids):
        return None, None, None

    onsets = [int(s.start) for s in spans]
    frame_sec = len(y) / SR / max(n_frames, 1)
    return onsets, n_frames, frame_sec


def stressed_window(record: dict, y_full: np.ndarray = None,
                    audio_root: Path = PROJECT_DIR, min_share: float = 0.10):
    """Time window of the stressed syllable for one dataset record.

    Pass `y_full` (the whole recording at 16 kHz) when looping over a file.
    Falls back to a proportional split of the word window when the acoustic
    alignment fails, so every record still yields a usable span; the `method`
    field records which one produced it.
    """
    raw = record.get("word_raw")
    letters = record["word_clean"]
    syl = stressed_syllable(raw) if raw else None

    if syl is None or not syl.get("ok"):
        return {"ok": False, "reason": "no vocalized form"}

    lo, hi = syl["letter_start"], syl["letter_end"]
    lo, hi = min(lo, len(letters)), min(hi, len(letters))
    if hi <= lo:
        return {"ok": False, "reason": "empty syllable span"}

    w_start, w_end = record["start"], record["end"]
    result = {
        "ok": True, "taam": record["taam"], "word": letters,
        "syllable": syl["text"], "rule": syl["rule"],
        "n_syllables": syl["n_syllables"], "from_end": syl["from_end"],
    }

    if y_full is None:
        path = audio_root / record["audio_file"]
        y_full = load_recording(path) if path.exists() else None

    if y_full is not None:
        y, offset = slice_window(y_full, w_start, w_end)
        onsets, n_frames, frame_sec = align_word(y, letters)
        if onsets and hi <= len(onsets):
            # A letter sounds from its own onset until the next letter's onset;
            # the final letter runs to the end of the window.
            s = offset + onsets[lo] * frame_sec
            e = offset + (onsets[hi] if hi < len(onsets) else n_frames) * frame_sec
            # Keep the window inside the word we were given: CTC may place a
            # letter in the padding, and a syllable outside its own word is a
            # failed alignment, not a short one.
            s, e = max(s, w_start), min(e, w_end)
            # Sung Hebrew stretches vowels far past anything in the model's
            # training data, and it sometimes answers by cramming the last
            # letters into the final frames. A stressed syllable that comes out
            # a sliver of its word is that failure, not a short syllable, so it
            # falls through to the proportional estimate instead of poisoning
            # the window.
            share = (e - s) / max(w_end - w_start, 1e-6)
            if e - s >= 0.03 and share >= min_share:
                result.update(start=round(s, 3), end=round(e, 3), method="ctc")
                return result
            result["ctc_rejected_share"] = round(share, 3)

    # Proportional fallback: letters are assumed evenly timed across the word.
    span = w_end - w_start
    n = max(len(letters), 1)
    result.update(start=round(w_start + span * lo / n, 3),
                  end=round(w_start + span * hi / n, 3),
                  method="proportional")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aligned_dataset_pt.json")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--demo", action="store_true", help="print each aligned word")
    ap.add_argument("--all", action="store_true", help="every record, ignore --limit")
    ap.add_argument("--write", metavar="PATH",
                    help="write the stressed windows to this json, keyed by "
                         "reader|parasha|start")
    args = ap.parse_args()

    with open(PROJECT_DIR / args.dataset, encoding="utf-8") as f:
        records = json.load(f)
    records = [r for r in records if r.get("word_raw")]
    if not args.all:
        records = records[: args.limit]
    written = {}

    # Group by recording so each file is decoded once.
    by_file = {}
    for r in records:
        by_file.setdefault(r["audio_file"], []).append(r)

    methods, ratios = Counter(), []
    per_taam, per_taam_ctc = {}, Counter()
    if args.demo:
        print(f"{'word':<16}{'taam':<16}{'syllable':<12}{'word window':>16}"
              f"{'stressed window':>18}{'share':>7}  method")
        print("-" * 92)

    # Checkpoint per recording. Aligning the whole corpus takes hours, and a run
    # that only saves at the end throws all of it away if it is interrupted.
    parts_dir = Path.home() / ".cache" / "taamify" / "ctc_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    for n_file, (audio_file, items) in enumerate(by_file.items(), 1):
        part = parts_dir / (audio_file.replace("/", "__") + ".json")
        if args.write and part.exists():
            try:
                with open(part, encoding="utf-8") as f:
                    cached = json.load(f)
                written.update(cached["windows"])
                for taam, shares in cached["shares"].items():
                    per_taam.setdefault(taam, []).extend(shares)
                    ratios.extend(shares)
                per_taam_ctc.update(cached["ctc_ok"])
                methods.update(cached["methods"])
                print(f"[{n_file}/{len(by_file)}] {Path(audio_file).name} "
                      f"(cached)", flush=True)
                continue
            except (json.JSONDecodeError, KeyError, OSError):
                pass          # corrupt checkpoint: just redo the recording

        path = PROJECT_DIR / audio_file
        y_full = load_recording(path) if path.exists() else None
        if args.all:
            print(f"[{n_file}/{len(by_file)}] {Path(audio_file).name} "
                  f"({len(items)} words)", flush=True)
        part_windows, part_shares, part_methods, part_ctc = {}, {}, Counter(), Counter()
        for r in items:
            out = stressed_window(r, y_full=y_full)
            if not out["ok"]:
                methods[out["reason"]] += 1
                part_methods[out["reason"]] += 1
                continue
            methods[out["method"]] += 1
            part_methods[out["method"]] += 1
            word_dur = r["end"] - r["start"]
            syl_dur = out["end"] - out["start"]
            share = syl_dur / word_dur if word_dur > 0 else 0
            ratios.append(share)
            per_taam.setdefault(out["taam"], []).append(share)
            per_taam_ctc[out["taam"]] += (out["method"] == "ctc")
            part_shares.setdefault(out["taam"], []).append(share)
            part_ctc[out["taam"]] += (out["method"] == "ctc")
            if args.write:
                key = f"{r['reader']}|{r['parasha']}|{r['start']}"
                entry = {"start": out["start"], "end": out["end"],
                         "method": out["method"]}
                written[key] = entry
                part_windows[key] = entry
            if args.demo:
                print(f"{out['word']:<16}{out['taam']:<16}{out['syllable']:<12}"
                      f"{r['start']:>7.2f}-{r['end']:<8.2f}"
                      f"{out['start']:>9.2f}-{out['end']:<8.2f}{share:>6.0%}  {out['method']}")

        if args.write:
            with open(part, "w", encoding="utf-8") as f:
                json.dump({"windows": part_windows, "shares": part_shares,
                           "methods": dict(part_methods), "ctc_ok": dict(part_ctc)}, f)

    print("-" * 92)
    n = sum(methods.values())
    print("methods:", {k: f"{v:,} ({100 * v / max(n, 1):.1f}%)" for k, v in methods.items()})
    if ratios:
        print(f"\nstressed syllable covers {np.mean(ratios):.1%} of the word window "
              f"on average (median {np.median(ratios):.1%})")
        print(f"→ the whole-word window feeds the network "
              f"{1 / max(np.mean(ratios), 1e-6):.1f}x more audio than carries the ta'am")

    if per_taam:
        print(f"\n{'ta_am':<18}{'n':>7}{'share':>8}{'ctc ok':>9}")
        print("-" * 42)
        for taam, shares in sorted(per_taam.items(), key=lambda kv: -len(kv[1])):
            ok = per_taam_ctc[taam]
            print(f"{taam:<18}{len(shares):>7,}{np.mean(shares):>7.0%}"
                  f"{100 * ok / len(shares):>8.0f}%")
        print("-" * 42)
        print("'ctc ok' = share of that ta'am whose window came from the acoustic\n"
              "alignment rather than the proportional fallback. A low value means\n"
              "the CTC model cannot follow that ta'am's melisma.")

    if args.write:
        out_path = PROJECT_DIR / args.write
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(written, f)
        print(f"\nwrote {len(written):,} stressed windows -> {out_path.name}")


if __name__ == "__main__":
    main()
