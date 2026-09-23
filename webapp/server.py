"""
Taamify web application — FastAPI backend
=========================================
Serves the single-page frontend (webapp/static/) and three API groups:

  GET  /api/info                     model + dataset facts for the header
  GET  /api/teacher                  practice catalogue (per-ta'am word clips)
  GET  /api/teacher/audio/{clip_id}  the teacher's audio clip
  POST /api/practice/{clip_id}       student WAV -> hybrid score + curves
  POST /api/verify_reading           student WAV + expected ta'am -> match/confidence
  POST /api/analyze                  full recording WAV -> per-word ta'am list

The browser records raw PCM and uploads standard WAV, so no ffmpeg is needed.
Feature extraction matches the training pipeline exactly.

/api/verify_reading attempts that come back no-match, or land in the
ambiguous confidence band, are logged locally (app_assets/collected_attempts/)
-- this is the only real path to a Stage B test set (the training corpus has
zero genuine reading errors; see results/v3_comparison/ben13_zeroshot_eval.md).
Local files only, nothing leaves the machine. A real pilot with other people's
voices needs a consent notice in the UI before this is turned on for them --
not built here, since there's no real user-facing deployment yet.

Run:  uvicorn webapp.server:app --port 8077        (from the project root)
"""

import importlib.util
import io
from functools import lru_cache
import json
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEACHER_DIR = PROJECT_DIR / "app_assets" / "teacher"
TEACHER_JSON = PROJECT_DIR / "app_assets" / "teacher_contours.json"
# Trained with hard negatives (the class the CNN itself confuses this one
# with), not random ones. Measured: a head trained on random negatives lets
# 19.7% of realistic learner mistakes through at the same contradiction rate;
# this one lets 9.9% through. A learner's error is never a random ta'am.
# NOTE: sweep jobs must COPY this file, never mv it -- a `mv` in a queued job
# once deleted the head out from under the running server.
VERIFY_HEAD_PATH = PROJECT_DIR / "results" / "verify_head_cal.pth"   # see note below
# Derived by Data_Prep/derive_neutral_band.py, which sets TWO thresholds
# against two different error budgets. The earlier file collapsed them to one
# number, which silently removed the neutral state the product depends on:
# every word was forced into "correct" or "wrong", including the ones the
# model had no real opinion about.
CALIBRATION_PATH = PROJECT_DIR / "results" / "calibration" / "calibration_served.json"
ATTEMPTS_DIR = PROJECT_DIR / "app_assets" / "collected_attempts"
ATTEMPTS_LOG = ATTEMPTS_DIR / "attempts_log.jsonl"
BEN13_SEGMENTS_DIR = PROJECT_DIR / "data" / "ben13_segments"
BEN13_SEGMENTS_MANIFEST = BEN13_SEGMENTS_DIR / "manifest.json"
BEN13_SOURCE_DIR = PROJECT_DIR / "data" / "ben13"
REVIEW_NOTES_PATH = BEN13_SEGMENTS_DIR / "review_notes.json"
SEPHARDI_NAMES_PATH = BEN13_SEGMENTS_DIR / "sephardi_names.json"
MANUAL_CUTS_PATH = BEN13_SEGMENTS_DIR / "manual_cuts.json"
WHISPER_CROSSCHECK_PATH = BEN13_SEGMENTS_DIR / "whisper_crosscheck.json"
ACOUSTIC_CHECK_PATH = BEN13_SEGMENTS_DIR / "acoustic_check.json"
BEN13_READING_META = PROJECT_DIR / "features_v3_word" / "BEN13R_meta.json"

# audio constants — must match the extractor that produced the features the
# LOADED checkpoint was trained on. The production model is now the Ben13
# Torah model, and Data_Prep/extract_ben13_reading_words.py builds its
# spectrograms at ctc_align.SR = 16000, not the 22050 the older
# extract_features_v2.py used for R1/R2/R8.
#
# This is not cosmetic. With librosa's default hop of 512, a frame is 32ms at
# 16k and 23.2ms at 22050, so the same 87 frames cover 2.78s versus 2.02s --
# every live spectrogram arrived time-compressed by ~38% relative to training.
# The classifier still often named the right ta'am, while the verify head
# (which is far more sensitive to the exact crop) collapsed: "בראשית" was
# heard correctly as Tipecha and still rejected at 0.24 confidence.
SR = 16000
N_MELS = 128
FMAX = 8000
MAX_FRAMES = 87
PAD_DB = -80.0

app = FastAPI(title="Taamify")

# ---------------------------------------------------------------- models --
_state = {}


HAFTARA_CKPT = (PROJECT_DIR /
                "results/random_top21_haftmix_FINAL/deep_cnn/deep_cnn_best.pth")


def get_haftara_model():
    """Separate model for haftara reading.

    The two genres gave OPPOSITE answers to the same question, measured
    head to head: adding haftara to the Torah model cost 7.8 points, while
    adding Torah to the haftara model gained 4.4. Haftara has only 25,214
    words -- too few to stand alone -- while the Torah's 64,649 are already
    enough and the haftara only adds noise. So: two models, and the app
    picks by what the learner is actually preparing.
    """
    if "haftara" not in _state:
        _state["haftara"] = None
        try:
            spec = importlib.util.spec_from_file_location(
                "tp", PROJECT_DIR / "10_train_pipeline.py")
            tp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tp)
            ck = torch.load(HAFTARA_CKPT, map_location="cpu", weights_only=False)
            model = tp.MODEL_REGISTRY[ck["architecture"]](
                num_classes=len(ck["class_names"]))
            model.load_state_dict(ck["model_state_dict"])
            st = get_model()
            model.eval().to(st["device"])
            _state["haftara"] = {"model": model,
                                 "class_names": list(ck["class_names"]),
                                 "device": st["device"]}
        except Exception as e:
            print(f"  haftara model unavailable ({e})")
    return _state["haftara"]


def get_model():
    if "model" not in _state:
        spec = importlib.util.spec_from_file_location(
            "tp", PROJECT_DIR / "10_train_pipeline.py")
        tp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tp)
        # 21 classes, not the original top-11. The 11-class model left 7.6% of
        # Torah words (5,243 of 68,613) with a ta'am the app simply could not
        # verify -- roughly 1 in 13 words a learner clicks. Trading a little
        # top-1 accuracy (78% -> 68.9% on a 21-way problem, which is a harder
        # problem, not a worse model) for coverage is the right call here,
        # because Stage A (the binary verify the product actually uses) got
        # *better*, not worse: 94.6% val vs 94.0%.
        # Ben13-only Torah model. The previous checkpoint was trained on the
        # three YouTube readers; measured head-to-head on the identical task,
        # adding those 80,285 words LOWERED accuracy by 4.3 points -- noisy
        # alignment does not average out, it smears over the clean data.
        # 87.9% on a random split, 86.9% with eight whole parashot held out
        # (the honest number for a single-voice corpus), and the verify head
        # trained on THIS checkpoint's embedding space scores 97.5%.
        ckpt_path = PROJECT_DIR / "results/parasha8_top21_torahhonest_FINALrecipe/deep_cnn/deep_cnn_best.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = tp.MODEL_REGISTRY[ckpt["architecture"]](
            num_classes=len(ckpt["class_names"]))
        model.load_state_dict(ckpt["model_state_dict"])
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.eval().to(device)
        _state.update(model=model, device=device,
                      class_names=ckpt["class_names"],
                      ckpt_path=str(ckpt_path),
                      test_acc=ckpt.get("test_metrics", {}).get("accuracy"))
    return _state


def get_verify_head():
    """Stage A (Milestone 3): [embedding ; one-hot(candidate class)] -> match
    logit, trained on top of this same production CNN's frozen embeddings
    (see 15_binary_verify_model.py, production mode). Loaded once, cached in
    _state like every other model here."""
    if "verify_head" not in _state:
        spec = importlib.util.spec_from_file_location(
            "bv", PROJECT_DIR / "15_binary_verify_model.py")
        bv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bv)
        ckpt = torch.load(VERIFY_HEAD_PATH, map_location="cpu", weights_only=False)
        st = get_model()
        if list(ckpt["class_names"]) != list(st["class_names"]):
            raise RuntimeError(
                "verify-head class_names don't match the loaded CNN's -- it "
                "was trained against a different checkpoint's embedding space")
        # Class names alone are not enough. Two runs of the same recipe produce
        # identical class lists and different embedding spaces, and on
        # 2026-09-15 the shipped head was found to have been trained against
        # random_top21_torahship_FINAL while the server was serving
        # random_top21_torahship -- no error, no 500, just quietly wrong
        # scores. The checkpoint path is what actually pins the pairing.
        trained_against = ckpt.get("cnn_checkpoint")
        # Resolve relative paths against the project, never the working
        # directory: the checkpoint stores "results/..." and a caller running
        # from Data_Prep/ resolved that to Data_Prep/results/..., firing this
        # guard on two names for the same file.
        def _abs(q):
            q = Path(q)
            return (q if q.is_absolute() else PROJECT_DIR / q).resolve()
        if trained_against and _abs(trained_against) != _abs(st["ckpt_path"]):
            raise RuntimeError(
                f"verify head was trained against {trained_against}, but the "
                f"server is serving {st['ckpt_path']} -- retrain the head "
                f"against the served checkpoint (15_binary_verify_model.py)")
        head = bv.VerifyHead(ckpt["emb_dim"], len(ckpt["class_names"]), ckpt["hidden"])
        head.load_state_dict(ckpt["model_state_dict"])
        head.eval().to(st["device"])
        _state["verify_head"] = head
    return _state["verify_head"]


def get_whisper():
    if "whisper" not in _state:
        from faster_whisper import WhisperModel
        _state["whisper"] = WhisperModel("small", device="cpu",
                                         compute_type="int8")
    return _state["whisper"]


def get_teacher():
    if "teacher" not in _state:
        with open(TEACHER_JSON, encoding="utf-8") as f:
            _state["teacher"] = json.load(f)
    return _state["teacher"]


def get_melody_curves():
    """Data_Prep/extract_melody_curves.py's output: real pitch contours
    (librosa.pyin, semitone-normalised, time-normalised to 60 points) for
    many independent words per ta'am across R1/R2/R8 -- so a reader can
    SEE whether a class's melody is actually stable, the same kind of
    check as Ben-Shalom/Keshet/Yeger-Granot (CIM14 2014)."""
    if "melody_curves" not in _state:
        path = PROJECT_DIR / "app_assets" / "melody_curves.json"
        with open(path, encoding="utf-8") as f:
            _state["melody_curves"] = json.load(f)
    return _state["melody_curves"]


@app.get("/api/melody")
def melody():
    curves = get_melody_curves()
    return {t: len(v) for t, v in curves.items()}


def get_class_medoids():
    """The single most *representative* real word per ta'am: the curve with
    the smallest total DTW distance to every other curve in its class (the
    medoid, not the mean -- a mean curve is an average of shapes and
    corresponds to no actual recording you can play).

    Why this matters for teaching: the existing teacher clips were picked
    arbitrarily (whichever word happened to come first in a chosen parasha,
    or Ben13's isolated demo). A medoid is defensible -- it is the recording
    that is least unusual for its class, and its distance is reportable."""
    if "class_medoids" not in _state:
        curves = get_melody_curves()
        out = {}
        for taam, items in curves.items():
            if len(items) < 3:
                continue
            mat = np.array([it["curve"] for it in items], dtype=float)
            n = len(mat)
            total = np.zeros(n)
            for i in range(n):
                for j in range(i + 1, n):
                    d = dtw_component(mat[i], mat[j])[1]
                    total[i] += d
                    total[j] += d
            best = int(np.argmin(total))
            it = items[best]
            out[taam] = {**it, "mean_distance": round(float(total[best] / (n - 1)), 3),
                         "n_compared": n}
        _state["class_medoids"] = out
    return _state["class_medoids"]


@app.get("/api/medoid/{taam}")
def medoid(taam: str):
    m = get_class_medoids().get(taam)
    if not m:
        return {"error": f"no representative example for {taam!r}"}
    return {k: v for k, v in m.items() if k != "audio_file"}


@app.get("/api/medoid/{taam}/audio")
def medoid_audio(taam: str):
    """The medoid word's actual audio, cut from the reader's recording."""
    import io as _io
    import soundfile as _sf
    m = get_class_medoids().get(taam)
    if not m:
        return {"error": "not found"}
    path = PROJECT_DIR / m["audio_file"]
    if not path.exists():
        return {"error": "source audio missing"}
    y, sr = librosa.load(str(path), sr=None, offset=max(0, m["start"] - 0.05),
                         duration=(m["end"] - m["start"]) + 0.1)
    buf = _io.BytesIO()
    _sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    from fastapi.responses import Response
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/api/melody/{taam}")
def melody_taam(taam: str):
    curves = get_melody_curves()
    if taam not in curves:
        return {"error": f"no melody data for {taam!r}"}
    return {"taam": taam, "curves": curves[taam]}


def get_ben13_source_index():
    """{(type_name, order): source-clip manifest entry} -- for showing the
    full original combo clip alongside a cut segment, so a reviewer can
    judge the cut in context rather than hearing it in isolation."""
    if "ben13_source_index" not in _state:
        idx = {}
        for type_dir in sorted(BEN13_SOURCE_DIR.iterdir()) if BEN13_SOURCE_DIR.exists() else []:
            mpath = type_dir / "manifest.json"
            if not mpath.exists():
                continue
            with open(mpath, encoding="utf-8") as f:
                for e in json.load(f):
                    idx[(type_dir.name, e["order"])] = e
        _state["ben13_source_index"] = idx
    return _state["ben13_source_index"]


def load_review_notes() -> dict:
    if not REVIEW_NOTES_PATH.exists():
        return {}
    with open(REVIEW_NOTES_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_review_note(clip_id: str, note: str, flag: str):
    notes = load_review_notes()
    if note or flag:
        notes[clip_id] = {"note": note, "flag": flag, "saved_at": time.time()}
    else:
        notes.pop(clip_id, None)
    REVIEW_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=1)


def _clean_hebrew(text):
    """Normalise the whitespace artifacts Ben13's source text carries.

    Two different things both need handling: a real U+00A0 non-breaking
    space, and -- the nastier one -- the four literal characters
    backslash-x-A-0, which are not whitespace at all. Python's str.split()
    treats the real nbsp as a separator but glues words together across the
    literal escape, so an entry like "מונח\\xA0רביעי" tokenizes as ONE word
    carrying TWO ta'amim, which is exactly the shape segment_ben13_clips.py
    rejects as unsplittable. Replacing it with a space first is what lets
    those entries tokenize correctly."""
    if not text:
        return text
    return text.replace("\\xA0", " ").replace("\\xa0", " ") \
               .replace("\xa0", " ").replace("\\n", " ").strip()


def get_ben13_last_regenerated() -> float:
    path = BEN13_SEGMENTS_DIR / "last_regenerated.json"
    if not path.exists():
        return 0.0
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("timestamp", 0.0)


def get_ben13_single_items() -> list:
    """Ben13 entries that demonstrate exactly one ta'am on their own -- these
    were only ever wired into the teacher-clip system (app_assets/teacher_
    contours.json), never into this review tool, even though they're 24 of
    Ben13's 75 clips and the only Ben13 coverage at all for 8 classes
    (Gershayim, Mercha_Kefula, Pazer, Shalshelet, Telisha_Gedola,
    Telisha_Qetana, Yetiv, Zaqef_Gadol never appear in any combo, so they
    never produced a segment). No cutting needed -- the whole clip already
    is the one ta'am -- so 'segment' and 'source' audio are the same file."""
    items = []
    for type_dir in sorted(BEN13_SOURCE_DIR.iterdir()) if BEN13_SOURCE_DIR.exists() else []:
        mpath = type_dir / "manifest.json"
        if not mpath.exists():
            continue
        with open(mpath, encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            if len(e["taam_sequence"]) != 1:
                continue
            audio_path = PROJECT_DIR / e["local_path"]
            if not audio_path.exists():
                continue
            items.append({
                "kind": "single",
                "clip_id": f"{e['taam_sequence'][0]}_{type_dir.name}-{e['order']}_solo",
                "taam": e["taam_sequence"][0],
                "source_type": type_dir.name,
                "source_order": e["order"],
                "ctc_score": None,
                "duration": None,
                "low_confidence": False,
                "label_note": None,
                "source_text": e.get("text"),
                "source_text_sfaradi": e.get("textSfaradi"),
                "_audio_path": audio_path,
            })
    return items


def load_manual_cuts() -> dict:
    if not MANUAL_CUTS_PATH.exists():
        return {}
    with open(MANUAL_CUTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_whisper_crosscheck() -> dict:
    """Data_Prep/whisper_crosscheck_ben13.py's output: an independent,
    unconstrained Whisper transcription of each combo clip, for comparison
    against Ben13's declared word count. Whisper routinely OVER-splits
    Sephardic compound names (e.g. "שופר-הולך" -> two ordinary Hebrew
    words), so an over-count is expected noise and not itself meaningful --
    only an UNDER-count (get_cut_clips filters to this) is a rare, useful
    signal that a declared word may not actually be sung, confirmed against
    the manually-found bad clips (order 180 both copies, order 20/100)."""
    if not WHISPER_CROSSCHECK_PATH.exists():
        return {}
    with open(WHISPER_CROSSCHECK_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_cut_clips() -> list:
    """One row per Ben13 *source* clip (not per segment) for the manual cut
    editor: the whole clip plus the current word boundaries, so a reviewer
    edits the transcript and drags the cuts in one place.

    Boundaries are seeded from whatever exists -- a saved manual cut wins,
    otherwise the CTC spans recorded by segment_ben13_clips.py -- so the
    reviewer corrects an existing guess rather than placing every handle
    from nothing."""
    manual = load_manual_cuts()
    whisper_xc = load_whisper_crosscheck()
    ctc_by_clip = {}
    if BEN13_SEGMENTS_MANIFEST.exists():
        with open(BEN13_SEGMENTS_MANIFEST, encoding="utf-8") as f:
            for seg in json.load(f):
                key = f"{seg['source_type']}-{seg['source_order']}"
                ctc_by_clip.setdefault(key, []).append(seg)

    # Third automatic signal, independent of the other two: average
    # seconds/word per clip. A clip whose declared text is right length but
    # wrong (e.g. describes different content, or a very low-score CTC run
    # compressed badly) doesn't necessarily fail the word-count checks --
    # but it does produce abnormally short segments, since the alignment is
    # squeezing real audio into too little of it. Confirmed against the
    # reviewer's own notes: clips flagged this way include ones with
    # complaints like "the cut sounds like the LAST word, not this one" --
    # a genuine bad alignment neither of the other two signals catches.
    ratios = {}
    for key, segs in ctc_by_clip.items():
        ctc_segs = [s for s in segs if s.get("cut_source", "ctc") == "ctc"]
        if len(ctc_segs) >= 2:
            ratios[key] = sum(s["duration"] for s in ctc_segs) / len(ctc_segs)
    _rv = sorted(ratios.values())
    _median_ratio = _rv[len(_rv) // 2] if _rv else 0
    short_keys = {k for k, v in ratios.items() if v < _median_ratio * 0.75}

    rows = []
    for type_dir in sorted(BEN13_SOURCE_DIR.iterdir()) if BEN13_SOURCE_DIR.exists() else []:
        mpath = type_dir / "manifest.json"
        if not mpath.exists():
            continue
        with open(mpath, encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            if len(e["taam_sequence"]) < 2:
                continue          # nothing to cut on a single-ta'am clip
            audio_path = PROJECT_DIR / e["local_path"]
            if not audio_path.exists():
                continue
            key = f"{type_dir.name}-{e['order']}"
            saved = manual.get(key)
            if saved:
                words = saved.get("words", [])
            else:
                segs = sorted(ctc_by_clip.get(key, []), key=lambda s: s.get("start") or 0)
                words = [{"label": _clean_hebrew(s.get("word_text", "")),
                          "taam": s["taam"],
                          "start": s.get("start"), "end": s.get("end")}
                         for s in segs if s.get("start") is not None]
            xc = whisper_xc.get(key)
            whisper_undercount = bool(xc and xc["whisper_n"] < xc["declared_n"])
            rows.append({
                "key": key,
                "source_type": type_dir.name,
                "source_order": e["order"],
                "declared_sequence": e["taam_sequence"],
                "text": _clean_hebrew(e.get("text")),
                "text_sfaradi": _clean_hebrew(e.get("textSfaradi")),
                "words": words,
                "is_manual": bool(saved),
                "text_fallback": any(s.get("text_fallback")
                                     for s in ctc_by_clip.get(key, [])),
                "low_confidence": any(s.get("low_confidence")
                                      for s in ctc_by_clip.get(key, [])),
                "whisper_undercount": whisper_undercount,
                "whisper_text": " ".join(xc["whisper_words"]) if xc else None,
                "short_duration": key in short_keys,
            })
    return rows


def save_manual_cut(key: str, words: list, note: str = ""):
    cuts = load_manual_cuts()
    if words:
        cuts[key] = {"words": words, "note": note, "saved_at": time.time()}
    else:
        cuts.pop(key, None)
    MANUAL_CUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_CUTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cuts, f, ensure_ascii=False, indent=1)


def load_sephardi_names() -> dict:
    if not SEPHARDI_NAMES_PATH.exists():
        return {}
    with open(SEPHARDI_NAMES_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_sephardi_name(taam: str, name: str):
    names = load_sephardi_names()
    if name:
        names[taam] = name
    else:
        names.pop(taam, None)
    SEPHARDI_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEPHARDI_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=1)


def load_acoustic_check() -> dict:
    """Data_Prep/segment_acoustic_check.py's per-segment output. Only the
    Whisper-silence half is trustworthy enough to surface (see that
    script's docstring -- the edge-truncation heuristic didn't separate
    clean cuts from bad ones on melismatic audio and was dropped, not
    shipped as a misleading badge)."""
    if not ACOUSTIC_CHECK_PATH.exists():
        return {}
    with open(ACOUSTIC_CHECK_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_review_items() -> list:
    """Every Ben13 segment (Data_Prep/segment_ben13_clips.py, Milestone 1C)
    plus its saved note/flag, if any. Read fresh each call -- these are small
    local JSON files, not worth caching and risking staleness against notes
    just saved a moment ago."""
    if not BEN13_SEGMENTS_MANIFEST.exists():
        return []
    with open(BEN13_SEGMENTS_MANIFEST, encoding="utf-8") as f:
        segments = json.load(f)
    notes = load_review_notes()
    source_idx = get_ben13_source_index()
    sephardi_names = load_sephardi_names()
    acoustic = load_acoustic_check()
    regenerated_at = get_ben13_last_regenerated()
    items = []
    for e in segments:
        clip_id = Path(e["local_path"]).stem
        source = source_idx.get((e["source_type"], e["source_order"]))
        n = notes.get(clip_id, {})
        saved_at = n.get("saved_at", 0.0)
        items.append({
            "kind": "segment",
            "clip_id": clip_id,
            "taam": e["taam"],
            "source_type": e["source_type"],
            "source_order": e["source_order"],
            "ctc_score": e["ctc_score"],
            "duration": e["duration"],
            "low_confidence": e["low_confidence"],
            "label_note": e.get("label_note"),
            "source_text": _clean_hebrew(source["text"]) if source else None,
            "source_text_sfaradi": _clean_hebrew(source["textSfaradi"]) if source else None,
            "note": n.get("note", ""),
            "flag": n.get("flag", ""),
            "stale_note": bool((n.get("note") or n.get("flag")) and regenerated_at
                               and saved_at < regenerated_at),
            "sephardi_name": sephardi_names.get(e["taam"], ""),
            "whisper_silent": acoustic.get(clip_id, {}).get("whisper_empty", False),
        })

    # Ben13's 24 single-ta'am clips (never cut, never entered this tool
    # before -- see get_ben13_single_items) round out coverage to 24 of the
    # project's 25 active classes (only Siluk has no Ben13 material at all,
    # in any form -- not something this tool or a code fix can add).
    for e in get_ben13_single_items():
        clip_id = e["clip_id"]
        n = notes.get(clip_id, {})
        saved_at = n.get("saved_at", 0.0)
        items.append({
            "kind": "single",
            "clip_id": clip_id,
            "taam": e["taam"],
            "source_type": e["source_type"],
            "source_order": e["source_order"],
            "ctc_score": None,
            "duration": None,
            "low_confidence": False,
            "label_note": None,
            "source_text": _clean_hebrew(e["source_text"]),
            "source_text_sfaradi": _clean_hebrew(e["source_text_sfaradi"]),
            "note": n.get("note", ""),
            "flag": n.get("flag", ""),
            "stale_note": False,
            "sephardi_name": sephardi_names.get(e["taam"], ""),
        })
    return items


# ------------------------------------------------------------ DSP helpers --
def word_spectrogram(y, start, end):
    s = max(0, int((start - 0.05) * SR))
    e = min(len(y), int((end + 0.05) * SR))
    if e <= s or (e - s) < SR * 0.1:
        return None
    mel = librosa.feature.melspectrogram(y=y[s:e], sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])),
                        mode="constant", constant_values=PAD_DB)
    return mel_db[:, :MAX_FRAMES].astype(np.float32)



SEQUENCE_TAGGER_PATH = (PROJECT_DIR / "results" / "sequence_model" / "calhonest_crf"
                        / "verse_tagger.pth")


def get_sequence_tagger():
    """Verse-level BiLSTM + CRF over the production CNN's frozen embeddings.

    This is the single biggest measured lever in the project and it was NOT
    reaching the product: /api/analyze classified each word on its own at
    87.8%, while the same words read in verse order score 94.6% -- the ta'amim
    are a punctuation system, so a word's neighbours genuinely constrain it.

    Returns None when the checkpoint is missing or was trained against a
    different CNN, so the per-word path stays available rather than the
    endpoint failing.
    """
    if "seq_tagger" not in _state:
        _state["seq_tagger"] = None
        try:
            if not SEQUENCE_TAGGER_PATH.exists():
                raise FileNotFoundError(SEQUENCE_TAGGER_PATH)
            spec = importlib.util.spec_from_file_location(
                "sm", PROJECT_DIR / "14_sequence_model.py")
            sm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sm)
            ck = torch.load(SEQUENCE_TAGGER_PATH, map_location="cpu",
                            weights_only=False)
            st = get_model()
            if list(ck["class_names"]) != list(st["class_names"]):
                raise RuntimeError(
                    "verse tagger was trained on a different CNN's classes")
            tagger = sm.VerseTagger(ck["in_dim"], len(ck["class_names"]))
            tagger.load_state_dict(ck["model_state_dict"])
            tagger.eval().to(st["device"])
            crf = None
            if "crf_state_dict" in ck:
                crf = sm.LinearChainCRF(len(ck["class_names"]))
                crf.load_state_dict(ck["crf_state_dict"])
                crf.eval().to(st["device"])
            _state["seq_tagger"] = {"tagger": tagger, "crf": crf,
                                    "class_names": ck["class_names"]}
        except Exception as e:
            print(f"  sequence tagger unavailable ({e}); "
                  f"/api/analyze falls back to per-word classification")
    return _state["seq_tagger"]


def tag_verse(specs):
    """Per-word ta'amim for one verse, read in order. Falls back to the
    per-word classifier when the tagger is unavailable."""
    sq = get_sequence_tagger()
    if sq is None or not specs:
        return None
    st = get_model()
    embs = np.stack([embed(sp) for sp in specs])
    x = torch.from_numpy(embs).float()[None].to(st["device"])
    lengths = torch.tensor([len(specs)])
    with torch.no_grad():
        emissions = sq["tagger"](x, lengths)
        if sq["crf"] is not None:
            mask = torch.ones(1, len(specs), dtype=torch.bool, device=x.device)
            path = sq["crf"].viterbi_decode(emissions, mask)[0]
        else:
            path = emissions[0].argmax(-1).cpu().tolist()
        probs = torch.softmax(emissions[0], dim=-1).cpu().numpy()
    return [{"taam": sq["class_names"][int(p)],
             "confidence": float(probs[i][int(p)])}
            for i, p in enumerate(path)]


def classify(spec, top_k=3, genre="torah"):
    st = get_haftara_model() if genre == "haftara" else None
    st = st or get_model()
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    x = torch.from_numpy(spec)[None, None].to(st["device"])
    with torch.no_grad():
        probs = torch.softmax(st["model"](x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1]
    return ([(st["class_names"][i], float(probs[i])) for i in order[:top_k]],
            {st["class_names"][i]: float(probs[i]) for i in range(len(probs))})


def embed(spec):
    """128-dim penultimate-layer embedding for one spectrogram, from the
    already-loaded production CNN -- same forward-hook technique as
    14_sequence_model.py's embed_everything(), just for a single live
    sample instead of a whole reader's worth of precomputed features."""
    st = get_model()
    last_linear = [m for m in st["model"].modules() if isinstance(m, nn.Linear)][-1]
    captured = []
    handle = last_linear.register_forward_hook(
        lambda _m, inp, _out: captured.append(inp[0].detach().cpu()))
    spec_norm = (spec - spec.mean()) / (spec.std() + 1e-8)
    x = torch.from_numpy(spec_norm)[None, None].to(st["device"])
    with torch.no_grad():
        st["model"](x)
    handle.remove()
    return captured[0][0].numpy()


def get_calibration():
    """Decision thresholds, measured on held-out parashot rather than guessed.

    The two mistakes cost different amounts. Telling a learner who read
    CORRECTLY that they were wrong costs trust, and a child who is
    contradicted while right stops believing the feedback. Staying quiet
    about a real mistake costs one uncorrected error in a practice session.
    So the threshold is chosen to hold contradictions under 2% and accept
    the misses that follow -- not to maximise accuracy.
    See Data_Prep/calibrate_served_path.py.

    A threshold only means anything against the score it was derived from, so
    the calibration file records the CONTEXT_WEIGHT in force when it was
    derived and this refuses to use it at a different weight. That pairing has
    already been broken twice -- once by deriving on the isolated-word score
    while serving the blended one, and once by retuning the weight after
    calibrating. Neither failed loudly; both just moved the operating point.
    """
    if "calibration" not in _state:
        default = {"say_wrong_below": 0.30, "say_correct_above": 0.30}
        try:
            loaded = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        w = loaded.get("context_weight")
        if w is not None and abs(w - CONTEXT_WEIGHT) > 1e-9:
            raise RuntimeError(
                f"{CALIBRATION_PATH.name} was derived at context_weight={w} "
                f"but the server runs CONTEXT_WEIGHT={CONTEXT_WEIGHT}. The "
                f"thresholds do not apply to this score -- re-run "
                f"Data_Prep/calibrate_served_path.py.")
        default.update(loaded)
        _state["calibration"] = default
    return _state["calibration"]


def verify_match(spec, expected_taam):
    """Stage A: does this spectrogram match expected_taam? Returns
    (is_match, confidence) -- confidence is the model's own sigmoid
    probability, not thresholded."""
    st = get_model()
    if expected_taam not in st["class_names"]:
        raise ValueError(f"unknown ta'am {expected_taam!r}")
    head = get_verify_head()
    emb = torch.from_numpy(embed(spec)).float()[None].to(st["device"])
    idx = st["class_names"].index(expected_taam)
    onehot = torch.zeros(1, len(st["class_names"]), device=st["device"])
    onehot[0, idx] = 1.0
    with torch.no_grad():
        prob = torch.sigmoid(head(emb, onehot)).item()
    return prob >= get_calibration()["say_wrong_below"], prob


def f0_curve(y, n_points=60):
    """pyin F0 -> key-normalized semitone curve (same as teacher extraction)."""
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR,
                            frame_length=1024, hop_length=256)
    good = ~np.isnan(f0)
    if good.sum() < 10:
        return None
    t = np.arange(len(f0))
    filled = np.interp(t, t[good], f0[good])
    semis = 12 * np.log2(filled / 55.0)
    semis -= np.median(semis[good])
    return np.interp(np.linspace(0, len(semis) - 1, n_points), t, semis)


def dtw_component(student, teacher):
    D, wp = librosa.sequence.dtw(np.atleast_2d(student), np.atleast_2d(teacher),
                                 metric="euclidean")
    cost = float(D[-1, -1] / len(wp))
    return float(np.clip(100 * (1.6 - cost) / 1.1, 0, 100)), cost


def load_wav(data: bytes):
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.nan_to_num(y)


AMBIGUOUS_BAND = (0.3, 0.7)


def log_attempt(audio_bytes: bytes, expected_taam: str, is_match: bool, confidence: float):
    """Save a no-match or ambiguous-confidence /api/verify_reading attempt
    locally -- the only real path to a Stage B test set, since the training
    corpus has zero genuine reading errors. A confident match teaches
    nothing new (that's most of the traffic); a no-match or a genuinely
    uncertain call is exactly the population Stage B eventually needs real
    ground truth for. Local files only; see the module docstring for the
    consent caveat before this is ever pointed at a real user."""
    if is_match and not (AMBIGUOUS_BAND[0] <= confidence <= AMBIGUOUS_BAND[1]):
        return
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    clip_id = uuid.uuid4().hex[:12]
    (ATTEMPTS_DIR / f"{clip_id}.wav").write_bytes(audio_bytes)
    record = {"clip_id": clip_id, "timestamp": round(time.time(), 3),
              "expected_taam": expected_taam, "is_match": is_match,
              "confidence": round(confidence, 3)}
    with open(ATTEMPTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



PHRASES_DIR = PROJECT_DIR / "app_assets" / "collected_phrases"
PHRASES_LOG = PHRASES_DIR / "phrases_log.jsonl"


def log_phrase(audio_bytes, words, taamim, result, voice_id, consent):
    """Keep a consented phrase recording, with the text it was read against.

    This is the growth engine for the corpus, and it is a different shape
    from log_attempt(): that one keeps only the uncertain single words,
    because it was built to find Stage B errors. A phrase recording is worth
    more than that. It arrives already paired with the exact words and
    ta'amim it was read against, so it needs no manual labelling -- the same
    CTC alignment that built the whole corpus turns it into training rows
    (Data_Prep/ingest_collected_phrases.py).

    Every phrase from a learner is therefore a new VOICE, which is the one
    thing 470 GPU-hours could not produce: the entire corpus is a single
    reader, and the model scores 15/16 on that reader and 0/2 on another.

    Stored only with explicit consent, only locally, and the voice id is a
    random handle rather than anything identifying.
    """
    if not consent:
        return None
    PHRASES_DIR.mkdir(parents=True, exist_ok=True)
    clip_id = uuid.uuid4().hex[:12]
    (PHRASES_DIR / f"{clip_id}.wav").write_bytes(audio_bytes)
    rec = {"clip_id": clip_id, "timestamp": round(time.time(), 3),
           "voice_id": voice_id or "anon", "words": words, "taamim": taamim,
           "aligned": bool(result.get("aligned")),
           "align_score": result.get("align_score"),
           "correct": result.get("correct"), "wrong": result.get("wrong"),
           "unsure": result.get("unsure"),
           "statuses": [w.get("status") for w in result.get("words", [])]}
    with open(PHRASES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return clip_id


# ------------------------------------------------------------------- API --
@app.get("/api/info")
def info():
    st = get_model()
    return {"model": "DeepCNN (production: best unseen-reader generalization)",
            "device": st["device"],
            "test_accuracy": st["test_acc"],
            "classes": st["class_names"]}


@app.get("/api/teacher")
def teacher_catalogue():
    by_taam = {}
    for clip_id, rec in get_teacher().items():
        by_taam.setdefault(rec["taam"], []).append({
            "clip_id": clip_id, "word": rec["word"],
            "verse_ref": rec["verse_ref"], "duration": rec["duration"],
            "curve": rec["f0_semitones"], "source": rec.get("source")})
    return by_taam


def get_ben13_reading_counts():
    """Per-class word counts from Ben13's reading corpus (64,649 words,
    100% ta'am-sequence match against Sefaria on every sampled verse --
    validate_ben13_reading.py) -- an independent, second real-audio source
    for the teaching layer to cite, not just the 11 isolated demo clips
    Milestone 1B pulled in."""
    if "ben13_reading_counts" not in _state:
        counts = {}
        if BEN13_READING_META.exists():
            with open(BEN13_READING_META, encoding="utf-8") as f:
                meta = json.load(f)
            for m in meta:
                counts[m["taam"]] = counts.get(m["taam"], 0) + 1
        _state["ben13_reading_counts"] = counts
    return _state["ben13_reading_counts"]


@app.get("/api/ben13_stats")
def ben13_stats():
    """Per-class counts from the Ben13 reading corpus, for a real citation
    line in the teaching layer -- not marketing copy, an actual number."""
    return get_ben13_reading_counts()


@app.get("/api/teacher/audio/{clip_id}")
def teacher_audio(clip_id: str):
    safe = "".join(c for c in clip_id if c.isalnum() or c == "_")
    return FileResponse(TEACHER_DIR / f"{safe}.wav", media_type="audio/wav")


@app.post("/api/practice/{clip_id}")
async def practice(clip_id: str, audio: UploadFile = File(...)):
    rec = get_teacher().get(clip_id)
    if rec is None:
        return {"error": "unknown clip"}
    raw_audio = await audio.read()
    y = load_wav(raw_audio)
    curve = f0_curve(y)
    if curve is None:
        return {"error": "no singing detected — sing louder and clearly"}
    teacher_curve = np.array(rec["f0_semitones"])
    dtw_score, cost = dtw_component(curve, teacher_curve)
    spec = word_spectrogram(y, 0, len(y) / SR)
    top3, prob_map = classify(spec) if spec is not None else ([], {})
    p_target = prob_map.get(rec["taam"], 0.0)
    # 60% classifier probability, 40% melodic DTW: a faithful imitation with
    # a typical (~0.5) single-clip CNN confidence still clears the praise bar
    combined = 60 * p_target + 0.40 * dtw_score
    return {"target": rec["taam"], "word": rec["word"],
            "score": round(combined, 1),
            "cnn_top3": [{"taam": t, "prob": round(p, 3)} for t, p in top3],
            "target_probability": round(p_target, 3),
            "dtw_component": round(dtw_score, 1),
            "dtw_cost_semitones": round(cost, 2),
            "student_curve": np.round(curve, 2).tolist(),
            "teacher_curve": np.round(teacher_curve, 2).tolist()}


@app.post("/api/verify_reading")
async def verify_reading(audio: UploadFile = File(...), expected_taam: str = Form(...)):
    """Milestone 3, Stage A: does this recording match expected_taam? Single
    word, not a whole verse -- matches exactly what Stage A was trained and
    measured on (results/v3_comparison/milestone3_binary_verify.md)."""
    raw = await audio.read()
    y = load_wav(raw)
    spec = word_spectrogram(y, 0, len(y) / SR)
    if spec is None:
        return {"error": "no audio detected — record louder and clearly"}
    try:
        is_match, confidence = verify_match(spec, expected_taam)
    except ValueError as e:
        return {"error": str(e)}
    log_attempt(raw, expected_taam, is_match, confidence)

    # The learner's own pitch line, plus the class's reference shape, so the
    # feedback can be "here is your melody against the expected one" instead
    # of a bare pass/fail. Same normalisation as the corpus curves
    # (Data_Prep/extract_melody_curves.py): 60 points, semitones relative to
    # this utterance's own median -- shape, not absolute pitch.
    student, reference = None, None
    try:
        c = f0_curve(y, n_points=60)
        if c is not None:
            student = [round(float(v), 3) for v in c]
    except Exception:
        student = None          # never let the drawing break the verdict
    try:
        curves = get_melody_curves().get(expected_taam) or []
        if curves:
            arr = np.array([cc["curve"] for cc in curves], dtype=float)
            reference = [round(float(v), 3) for v in arr.mean(axis=0)]
    except Exception:
        reference = None

    return {"is_match": is_match, "confidence": round(confidence, 3),
            "expected_taam": expected_taam,
            "student_curve": student, "reference_curve": reference}




# ── extra channels: stress, dagesh, gutturals ────────────────────────────
# Each is an independent CNN over the SAME 128x87 word spectrogram, trained
# on the same corpus with a different label set. Measured on held-out
# parashot (balanced accuracy, trivial baseline 50%):
#   stress (milra vs mil'el)  93.5%
#   dagesh kal (b/v, k/kh)    93.1%
#   ayin vs alef              95.1%
# The expected answer for each comes from the POINTED TEXT, not from a model,
# so a mismatch is a real reading error rather than two models disagreeing.
CHANNELS = {
    "stress":   {"dir": "results/stress8_stress_PROD",
                 "classes": ["milra", "milel"], "he": "הטעמה"},
    "dagesh":   {"dir": "results/dagesh8_p_PROD",
                 "classes": ["no_dagesh", "dagesh"], "he": "דגש קל"},
    "guttural": {"dir": "results/guttural8_p_PROD",
                 "classes": ["alef", "ayin"], "he": "ע׳ / א׳"},
    # Median of three seeds (0.9616 / 0.9592 / 0.9601 balanced) -- the spread
    # is 0.24 points, far inside the 1.17-point seed noise, so picking the
    # highest would be cherry-picking. Shuffled-label control: 0.493.
    "sheva":    {"dir": "results/sheva8_p_PROD",
                 "classes": ["sheva_nach", "sheva_na"], "he": "שווא נע/נח"},
}
BEGADKEFAT = set("בגדכפת")
DAGESH_MARK = "\u05bc"


def get_channel(name):
    """Load one extra-channel CNN, cached. Returns None when the checkpoint is
    missing, so a channel can be absent without taking the endpoint down."""
    key = f"channel_{name}"
    if key not in _state:
        _state[key] = None
        try:
            cfg = CHANNELS[name]
            ck_path = next((PROJECT_DIR / cfg["dir"]).glob("*/*_best.pth"))
            spec = importlib.util.spec_from_file_location(
                "tp", PROJECT_DIR / "10_train_pipeline.py")
            tp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tp)
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            model = tp.MODEL_REGISTRY[ck["architecture"]](
                num_classes=len(ck["class_names"]))
            model.load_state_dict(ck["model_state_dict"])
            st = get_model()
            model.eval().to(st["device"])
            _state[key] = {"model": model, "classes": list(ck["class_names"]),
                           "he": cfg["he"]}
        except Exception as e:
            print(f"  channel {name} unavailable ({e})")
    return _state[key]


def hebrew_letters(w):
    return [c for c in w if "\u05d0" <= c <= "\u05ea"]


def expected_channel_label(name, pointed_word):
    """What the TEXT requires for this channel, or None when the word cannot
    carry the distinction at all -- a word with two begadkefat letters gives
    an ambiguous whole-word label, so it is skipped rather than guessed."""
    if not pointed_word:
        return None
    if name == "stress":
        try:
            spec = importlib.util.spec_from_file_location(
                "ss", PROJECT_DIR / "Data_Prep" / "stressed_syllable.py")
            ss = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ss)
            r = ss.stressed_syllable(pointed_word)
        except Exception:
            return None
        if not r or not r.get("ok") or r["n_syllables"] < 2:
            return None
        return {1: "milra", 2: "milel"}.get(r["from_end"])
    L = hebrew_letters(pointed_word)
    if name == "dagesh":
        b = [c for c in L if c in BEGADKEFAT]
        if len(b) != 1:
            return None
        i = pointed_word.index(b[0])
        return "dagesh" if DAGESH_MARK in pointed_word[i:i + 3] else "no_dagesh"
    if name == "guttural":
        a = [c for c in L if c in "\u05e2\u05d0"]
        if len(a) != 1:
            return None
        return "ayin" if a[0] == "\u05e2" else "alef"
    if name == "sheva":
        # sheva.py returns None wherever the pointing alone cannot settle it
        # -- after a qamats, which needs morphology. Those words are skipped,
        # never guessed: a confident wrong verdict is worse than silence.
        try:
            spec = importlib.util.spec_from_file_location(
                "shv", PROJECT_DIR / "Data_Prep" / "sheva.py")
            shv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shv)
            lab = shv.word_label(pointed_word)
        except Exception:
            return None
        return {"na": "sheva_na", "nach": "sheva_nach"}.get(lab)
    return None


def run_channels(spec_img, pointed_word, wanted=None):
    """Every extra channel that applies to this word. Silent on the ones that
    do not apply, rather than reporting a meaningless verdict."""
    out = {}
    for name in (wanted or CHANNELS):
        expected = expected_channel_label(name, pointed_word)
        if expected is None:
            continue
        ch = get_channel(name)
        if ch is None:
            continue
        st = get_model()
        x = ((spec_img - spec_img.mean()) / (spec_img.std() + 1e-8))
        t = torch.from_numpy(x)[None, None].to(st["device"])
        with torch.no_grad():
            probs = torch.softmax(ch["model"](t), dim=1)[0].cpu().numpy()
        heard = ch["classes"][int(probs.argmax())]
        conf = float(probs.max())
        out[name] = {"expected": expected, "heard": heard,
                     "status": "ok" if heard == expected else "wrong",
                     "confidence": round(conf, 3), "he": ch["he"]}
    return out



def get_severity():
    """Data_Prep/severity.py -- how serious each kind of mistake is.

    Until this was wired in, every channel reported a flat right/wrong, which
    asserts that all errors weigh the same. They do not: an error that
    changes the meaning is corrected, one that does not draws a rebuke. On
    2,500 held-out words only 19% of the model's ta'am errors touch a class
    that can change meaning -- most of its weakness sits in the conjunctives,
    where swapping one for another moves no boundary at all."""
    if "severity" not in _state:
        spec = importlib.util.spec_from_file_location(
            "sev", PROJECT_DIR / "Data_Prep" / "severity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _state["severity"] = mod
    return _state["severity"]



def context_probabilities(specs):
    """Per-word probability over the ta'amim, given the WHOLE phrase.

    verify_match() judges a word in isolation, which throws away the fact
    that the ta'amim are a punctuation system: a word's neighbours constrain
    it. Measured, that context is worth +6.8 points on classification
    (87.8% -> 94.6%) -- and it was reaching only /api/analyze, never the
    verification decision the product actually runs.

    It matters most exactly where the model is weakest. Of the five biggest
    confusions, three are between conjunctives (Mercha/Munach/Qadma) which
    serve different hosts and are separated by position rather than by sound,
    and one is Etnachta/Segol, where the grammar already knows a verse has
    exactly one etnachta.

    Returns None when the tagger is unavailable, so verification degrades to
    the per-word answer rather than failing.
    """
    sq = get_sequence_tagger()
    if sq is None or not specs:
        return None
    st = get_model()
    try:
        embs = np.stack([embed(sp) for sp in specs])
        x = torch.from_numpy(embs).float()[None].to(st["device"])
        with torch.no_grad():
            emissions = sq["tagger"](x, torch.tensor([len(specs)]))
            probs = torch.softmax(emissions[0], dim=-1).cpu().numpy()
    except Exception:
        return None
    names = sq["class_names"]
    return [{names[j]: float(row[j]) for j in range(len(names))} for row in probs]


# Measured, not guessed: eval_context_verify.py sweeps this against a fixed
# 2% contradict-a-correct-reader budget. On the honest chain (parasha-split
# CNN, held-out parashot) the word alone catches 55.6% of real wrong
# readings and 0.7 catches 74.5% -- +18.9 points.
#
# The first sweep, against the leaky random-split CNN, said 0.4 and +9.2. It
# understated the gain because the per-word score was itself inflated: the
# CNN had seen neighbouring words of the very verses being scored, so there
# was less for context to add. Re-measuring on the honest chain moved both
# the optimum and its size.
CONTEXT_WEIGHT = 0.7


def combine_verdict(word_prob, ctx_prob):
    """Blend the isolated-word verify score with what the phrase implies.

    A geometric mean rather than an average: both signals have to be
    unhappy before the word is called wrong, which keeps the expensive
    error -- contradicting a reader who was right -- rare. CONTEXT_WEIGHT
    sets how much the phrase is allowed to speak.
    """
    if ctx_prob is None:
        return word_prob
    w = max(word_prob, 1e-6) ** (1 - CONTEXT_WEIGHT)
    c = max(ctx_prob, 1e-6) ** CONTEXT_WEIGHT
    return float(w * c)



@lru_cache(maxsize=1)
def get_per_class_thresholds():
    """Per-ta'am say-wrong thresholds, calibrated against per-ta'am severity.

    One global threshold forces the same caution everywhere, but the ta'amim
    are not equally confusable and an error does not cost the same on each.
    Etnachta is almost never mistaken and changes the meaning of the verse, so
    it is worth being strict there; Mercha and Munach are separated mostly by
    position rather than by sound, so a confident verdict on one of them
    contradicts readers who were right.

    Falls back to the global threshold for any class not calibrated.
    """
    path = PROJECT_DIR / "results" / "calibration" / "per_class.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return {k: v["say_wrong_below"] for k, v in json.load(f).items()}
    except Exception:
        return {}


def decide(conf, expected=None):
    """The verdict, in one place.

    Everything that reports a number about this product -- the live endpoint,
    the calibration script, the Stage B evaluation -- has to run this same
    function, or the thresholds get derived against one score and applied to
    another. That has already happened once: the band was calibrated on the
    isolated-word score while the endpoint served the context-blended one.
    """
    cal = get_calibration()
    lo = get_per_class_thresholds().get(expected, cal["say_wrong_below"])
    hi = max(cal["say_correct_above"], lo)
    if conf >= hi:
        return "ok"
    if conf < lo:
        return "wrong"
    return "unsure"


def score_word(spec, expected, ctx_prob=None):
    """The confidence the product acts on: the isolated word, blended with
    what the phrase implies when a phrase is available."""
    _, raw = verify_match(spec, expected)
    return combine_verdict(raw, ctx_prob), raw



# The corpus's median word window is 1.10s, because Ben13 is recorded one
# word per file and the corpus took whole files. Matching that geometry at
# serving time matters more than matching the word's true extent. Measured on
# real held-out audio, same words, same model:
#
#   tight aligner cut            14.1%
#   partition the audio          62.5%
#   partition + 0.3s lead-in     65.6%
#   fixed 1.10s window           78.1%   <- this
#
# against 86.2% on the corpus's own windows. The window is a stand-in for the
# recording the model was trained on, not a transcription boundary.
CORPUS_WINDOW_S = 1.10
CORPUS_LEAD_IN_S = 0.25


def partition_spans(spans, duration):
    """Turn the aligner's tight word extents into a partition of the audio.

    The aligner marks where a word's consonants and vowels are -- 0.24s for
    a word like לֵךְ. The corpus the model was trained on did something very
    different: Ben13's recordings are one word per file, and the corpus took
    the WHOLE file as that word's window, melismatic tail, leading silence
    and all. Median corpus window: 1.10s. So the model has never seen a tight
    cut, and feeding it one hands it a window that is mostly padding.

    Measured on real held-out audio: tight cuts classify at 14.1%, the same
    words partitioned at 62.5%. The model was never the problem.

    So each word is given everything up to the midpoint of the gap to its
    neighbour, the first word keeps the leading silence and the last keeps
    the tail -- which is what a one-word recording looks like.
    """
    out = {}
    for k, (a, _b) in spans.items():
        lo = max(0.0, a - CORPUS_LEAD_IN_S)
        out[k] = (lo, min(duration, lo + CORPUS_WINDOW_S))
    return out


@app.post("/api/check_phrase")
async def check_phrase(audio: UploadFile = File(...),
                       words: str = Form(...),
                       taamim: str = Form(...),
                       channels: str = Form(""),
                       genre: str = Form("torah"),
                       consent: str = Form(""),
                       voice_id: str = Form("")):
    """The learning loop's core call: a recording of one phrase, plus the
    words and ta'amim the text requires, in order. Returns a verdict per
    word.

    This is Stage A applied phrase by phrase rather than to a single word.
    The learner never sings one word alone -- doing so changes the timing and
    the melody -- so the recording is aligned to the expected words first
    (the same CTC aligner the whole corpus was built with), and each word is
    then verified against the ta'am the text demands. A word the aligner
    cannot place is reported as such rather than guessed at, because telling
    a child "wrong" when the system simply lost the word is the most costly
    error this product can make.
    """
    wanted = [c.strip() for c in channels.split(",") if c.strip() in CHANNELS] or None
    # words/taamim arrive as JSON arrays. Malformed input is the caller's
    # mistake and belongs in a 400 with a usable message, not a 500 -- the
    # app has to be able to tell "you sent the wrong shape" apart from "the
    # model broke".
    try:
        parsed_words = json.loads(words)
        parsed_taamim = json.loads(taamim)
        if not isinstance(parsed_words, list) or not isinstance(parsed_taamim, list):
            raise ValueError("expected JSON arrays")
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"words and taamim must be JSON arrays of strings ({exc})")
    word_list = [w for w in parsed_words if w.strip()]
    taam_list = parsed_taamim
    if len(word_list) != len(taam_list):
        return {"error": "words and taamim must be the same length"}

    raw_audio = await audio.read()
    y = load_wav(raw_audio)
    if y is None or len(y) < SR * 0.2:
        return {"error": "ההקלטה קצרה מדי"}

    spec_mod = importlib.util.spec_from_file_location(
        "va", PROJECT_DIR / "Data_Prep" / "verse_align.py")
    va = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(va)
    ca = importlib.util.spec_from_file_location(
        "cv", PROJECT_DIR / "Data_Prep" / "clean_align_v2.py")
    cv = importlib.util.module_from_spec(ca)
    ca.loader.exec_module(cv)

    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    prepared = [{"raw": w, "norm": cv.normalize_word(w)} for w in word_list]
    try:
        spans, score, _ = va.align_verse(y16, prepared)
    except Exception as e:
        return {"error": f"alignment failed: {e}"}
    if spans is None:
        return {"aligned": False,
                "message": "לא הצלחתי ליישר את ההקלטה לטקסט — נסה להקליט שוב",
                "words": []}

    spans = partition_spans(spans, len(y16) / 16000)

    # first pass: cut every word, so the phrase can be judged as a sequence
    specs_in_order, spec_index = [], {}
    for i in range(len(word_list)):
        if i in spans:
            st_, en = spans[i]
            sp = word_spectrogram(y, st_ * SR / 16000, en * SR / 16000)
            if sp is not None:
                spec_index[i] = len(specs_in_order)
                specs_in_order.append(sp)
    ctx = context_probabilities(specs_in_order)

    results, correct, checked = [], 0, 0
    for i, (w, expected) in enumerate(zip(word_list, taam_list)):
        if i not in spans or not expected:
            results.append({"word": w, "expected": expected, "status": "unknown",
                            "message": "לא נמצאה במיקום ברור בהקלטה"})
            continue
        st_, en = spans[i]
        # the aligner works at 16k, the feature extractor at SR
        spec = word_spectrogram(y, st_ * SR / 16000, en * SR / 16000)
        if spec is None:
            results.append({"word": w, "expected": expected, "status": "unknown",
                            "message": "המקטע קצר מדי לניתוח"})
            continue
        st = get_model()
        if expected not in st["class_names"]:
            results.append({"word": w, "expected": expected, "status": "unsupported",
                            "message": "הטעם הזה עדיין לא נתמך"})
            continue
        ctx_p = None
        if ctx is not None and i in spec_index:
            ctx_p = ctx[spec_index[i]].get(expected)
        conf, raw_conf = score_word(spec, expected, ctx_p)
        # Three states, not two. Between the thresholds the model has no real
        # opinion, and the honest answer is a neutral mark rather than a
        # verdict -- contradicting a learner who read correctly is the one
        # mistake this product cannot afford to make often.
        status = decide(conf, expected)
        checked += 1
        correct += 1 if status == "ok" else 0
        top3, _ = classify(spec, genre=genre)
        extra = run_channels(spec, w, wanted)
        sev = get_severity()
        levels = []
        if status == "wrong":
            v = sev.verdict_for("taam", expected=expected,
                                heard=top3[0][0] if top3 else None)
            levels.append(v["level"])
            taam_note = v
        else:
            taam_note = None
        for name, ch in extra.items():
            if ch["status"] == "wrong":
                cv = sev.verdict_for(name)
                ch["severity"] = cv["level"]
                ch["severity_he"] = cv["level_he"]
                ch["message"] = cv["message"]
                levels.append(cv["level"])
        results.append({
            "severities": levels,
            "taam_note": taam_note,
            "word": w, "expected": expected,
            "status": status,
            "channels": extra,
            "confidence": round(float(conf), 3),
            "confidence_word_only": round(float(raw_conf), 3),
            "confidence_context": round(float(ctx_p), 3) if ctx_p is not None else None,
            "heard": top3[0][0] if top3 else None,
            "start": round(st_, 3), "end": round(en, 3),
        })
    ch_summary = {}
    for r in results:
        for name, v in (r.get("channels") or {}).items():
            c = ch_summary.setdefault(name, {"he": v["he"], "checked": 0, "correct": 0})
            c["checked"] += 1
            c["correct"] += 1 if v["status"] == "ok" else 0
    wrong_n = sum(1 for r in results if r.get("status") == "wrong")
    unsure_n = sum(1 for r in results if r.get("status") in ("unsure", "unknown"))
    return {"aligned": True,
            "align_score": round(float(score), 3) if score is not None else None,
            "checked": checked, "correct": correct,
            "wrong": wrong_n, "unsure": unsure_n,
            "channel_summary": ch_summary,
            "severity": get_severity().summarise(results),
            "genre": genre,
            "used_context": ctx is not None,
            "saved_clip": log_phrase(raw_audio, word_list, taam_list,
                                     {"aligned": True, "align_score": score,
                                      "correct": correct, "wrong": wrong_n,
                                      "unsure": unsure_n, "words": results},
                                     voice_id, consent.lower() in ("1", "true", "yes")),
            "thresholds": get_calibration(),
            "accuracy": round(correct / checked, 3) if checked else None,
            "words": results}


@app.get("/api/genres")
def genres():
    """Which reading genres the server can score. The app should send the one
    the learner picked; sending the wrong one costs real accuracy, because
    the two models were trained on deliberately different mixes."""
    return {"torah": {"available": True, "he": "תורה",
                      "honest_accuracy": 0.869},
            "haftara": {"available": get_haftara_model() is not None,
                        "he": "הפטרה", "honest_accuracy": 0.783}}


@app.get("/api/verify_config")
def verify_config():
    """What the confidence numbers in /api/check_phrase actually mean.

    Exposed so the UI can draw its own line instead of hardcoding one. Every
    rate below was measured on eight whole parashot the model never trained
    on, with mistakes built the way learners actually make them -- the ta'am
    the classifier most confuses the right one with, not a random ta'am.
    """
    cal = get_calibration()
    return {
        "say_wrong_below": cal["say_wrong_below"],
        "say_correct_above": cal["say_correct_above"],
        "measured": {
            "contradicts_correct_rate": cal.get("contradicts_correct_rate"),
            "lets_confusable_mistake_through": cal.get("lets_confusable_mistake_through"),
            "held_out_parashot": cal.get("held_out_parashot"),
            "n_positives": cal.get("n_positives"),
        },
        "note": ("below say_wrong_below -> safe to tell the learner they were "
                 "wrong; at or above say_correct_above -> safe to confirm; "
                 "between the two -> show a neutral mark and no verdict. When "
                 "the two are equal the measured distributions did not overlap "
                 "and no neutral zone is needed."),
    }


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    y = load_wav(await audio.read())
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    segments, _ = get_whisper().transcribe(
        y16, language="he", word_timestamps=True, beam_size=5)
    words = []
    pending = []          # (word_text, spec) in reading order, for the tagger
    for seg in segments:
        for w in (seg.words or []):
            spec = word_spectrogram(y, w.start, w.end)
            if spec is None:
                continue
            pending.append((w.word.strip(), spec))
            top3, _ = classify(spec)
            words.append({"word": w.word.strip(),
                          "start": round(w.start, 2), "end": round(w.end, 2),
                          "predictions": [{"taam": t, "prob": round(p, 3)}
                                          for t, p in top3]})
    # Second pass over the same words, this time in order and with verse
    # context. Measured 87.8% -> 94.6% on held-out parashot, so the per-word
    # answer is kept only as a fallback and for comparison.
    seq = tag_verse([sp for _, sp in pending])
    if seq and len(seq) == len(words):
        for w, s_ in zip(words, seq):
            preds = w.get("predictions") or []
            w["per_word_taam"] = preds[0]["taam"] if preds else None
            w["taam"] = s_["taam"]
            w["confidence"] = s_["confidence"]
        return {"words": words, "method": "verse_context_crf"}
    return {"words": words, "method": "per_word"}


# ------------------------------------------------------- Ben13 review UI --
# Internal QA tool (webapp/static/review.html), not part of the practice
# app: browse every Ben13 segment cut by Data_Prep/segment_ben13_clips.py
# (Milestone 1C) and record a note on each, so a reviewer isn't limited to
# manually-sent one-off files.
def safe_clip_id(clip_id: str) -> str:
    return "".join(c for c in clip_id if c.isalnum() or c in "_-")


@app.get("/api/review/items")
def review_items():
    return get_review_items()


@app.get("/api/review/audio/{clip_id}")
def review_audio(clip_id: str):
    clip_id = safe_clip_id(clip_id)
    for e in get_review_items():
        if e["clip_id"] == clip_id:
            if e["kind"] == "single":
                # never cut -- the whole Ben13 clip already is the one
                # ta'am, so "segment" and "source" audio are the same file.
                source = get_ben13_source_index().get((e["source_type"], e["source_order"]))
                if source:
                    p = PROJECT_DIR / source["local_path"]
                    if p.exists():
                        return FileResponse(p, media_type="audio/mpeg")
                break
            for type_dir in BEN13_SEGMENTS_DIR.iterdir() if BEN13_SEGMENTS_DIR.exists() else []:
                p = type_dir / f"{clip_id}.wav"
                if p.exists():
                    return FileResponse(p, media_type="audio/wav")
    return {"error": "clip not found"}


@app.get("/api/review/source_audio/{clip_id}")
def review_source_audio(clip_id: str):
    """The full original combo clip a segment was cut from -- context for
    judging whether the cut point itself was right."""
    clip_id = safe_clip_id(clip_id)
    for e in get_review_items():
        if e["clip_id"] == clip_id:
            source = get_ben13_source_index().get((e["source_type"], e["source_order"]))
            if source:
                p = PROJECT_DIR / source["local_path"]
                if p.exists():
                    return FileResponse(p, media_type="audio/mpeg")
    return {"error": "source clip not found"}


@app.post("/api/review/note")
def review_note(payload: dict = Body(...)):
    clip_id = safe_clip_id(payload.get("clip_id", ""))
    if not clip_id:
        return {"error": "missing clip_id"}
    save_review_note(clip_id, payload.get("note", ""), payload.get("flag", ""))
    return {"ok": True}


# ------------------------------------------------------ ta'am grammar --
# The teaching layer (Data_Prep/taam_grammar.py): the accents encode a
# recursive division of the verse, so the app can answer "why does the
# verse break here?" rather than only "wrong ta'am". Parses 100% of the
# 5,846 cached Torah verses.
def get_taam_grammar():
    if "taam_grammar" not in _state:
        spec = importlib.util.spec_from_file_location(
            "taam_grammar", PROJECT_DIR / "Data_Prep" / "taam_grammar.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _state["taam_grammar"] = mod
    return _state["taam_grammar"]



# ── practice plan: parasha -> aliya -> verse -> phrase ───────────────────
BEN13_MAINTEXT = PROJECT_DIR / "data" / "ben13_reading" / "maintext.json"
BOOK_NAMES = {"1": "Genesis", "2": "Exodus", "3": "Leviticus",
              "4": "Numbers", "5": "Deuteronomy"}
PARASHA_NAMES = [
    "בראשית", "נח", "לך לך", "וירא", "חיי שרה", "תולדות", "ויצא", "וישלח",
    "וישב", "מקץ", "ויגש", "ויחי", "שמות", "וארא", "בא", "בשלח", "יתרו",
    "משפטים", "תרומה", "תצוה", "כי תשא", "ויקהל", "פקודי", "ויקרא", "צו",
    "שמיני", "תזריע", "מצורע", "אחרי מות", "קדושים", "אמור", "בהר", "בחקתי",
    "במדבר", "נשא", "בהעלתך", "שלח", "קרח", "חקת", "בלק", "פינחס", "מטות",
    "מסעי", "דברים", "ואתחנן", "עקב", "ראה", "שופטים", "כי תצא", "כי תבוא",
    "נצבים", "וילך", "האזינו", "וזאת הברכה",
]


def get_practice_index():
    """parasha -> aliya -> ordered verse refs, from Ben13's own manifest.

    Ben13 already carries the aliya boundaries a reader actually needs, so
    the practice plan follows the real liturgical division rather than a
    chapter count the learner will never be asked to read.
    """
    if "practice_index" not in _state:
        idx = {}
        if BEN13_MAINTEXT.exists():
            with open(BEN13_MAINTEXT, encoding="utf-8") as f:
                rows = json.load(f)
            for r in rows:
                par, ali, book = r.get("parasha"), r.get("aliya"), r.get("book")
                if not par or not ali or book not in BOOK_NAMES:
                    continue
                if str(r.get("block")) in ("-1", "0"):
                    continue
                ref = (BOOK_NAMES[book], int(r["perek"]), int(r["pasuk"]))
                idx.setdefault(par, {}).setdefault(ali, [])
                if ref not in idx[par][ali]:
                    idx[par][ali].append(ref)
        _state["practice_index"] = idx
    return _state["practice_index"]


@app.get("/api/parashot")
def parashot():
    idx = get_practice_index()
    out = []
    for i, name in enumerate(PARASHA_NAMES, start=1):
        aliyot = idx.get(str(i), {})
        if not aliyot:
            continue
        out.append({
            "id": str(i), "name": name,
            "aliyot": [{"aliya": a, "verses": len(v)}
                       for a, v in sorted(aliyot.items(), key=lambda kv: int(kv[0]))],
            "total_verses": sum(len(v) for v in aliyot.values()),
        })
    return {"parashot": out}


def phrase_split(tokens):
    """Split a verse at each disjunctive accent -- the same rule Ben13's own
    segmentation follows, and the division the ta'amim themselves encode.
    Practice units are therefore the text's own phrases, not arbitrary
    chunks of N words."""
    tg = get_taam_grammar()
    out, start = [], 0
    for i, (_, taam) in enumerate(tokens):
        if taam in tg.DISJUNCTIVE_RANK:
            out.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        out.append((start, len(tokens)))
    return out


@app.get("/api/practice_plan")
def practice_plan(parasha: str, aliya: str):
    """Everything needed to practise one aliya: its verses, each split into
    phrases, each phrase carrying its words and their expected ta'amim."""
    tg = get_taam_grammar()
    idx = get_practice_index()
    refs = idx.get(parasha, {}).get(aliya, [])
    if not refs:
        return {"error": f"no data for parasha {parasha} aliya {aliya}"}
    name = PARASHA_NAMES[int(parasha) - 1] if parasha.isdigit() else parasha
    verses, n_phrases, n_words = [], 0, 0
    for book, chap, vnum in refs:
        path = PROJECT_DIR / "Data_Prep" / "sefaria_cache" / f"{book}.{chap}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if not 0 < vnum <= len(cached):
            continue
        raw = cached[vnum - 1]
        tokens = tg.verse_tokens(raw)
        if not tokens:
            continue
        phrases = []
        for pi, (lo, hi) in enumerate(phrase_split(tokens)):
            words = [{"word": w, "taam": t,
                      "taam_he": tg.TAAM_HE.get(t, t) if t else None}
                     for w, t in tokens[lo:hi]]
            phrases.append({"index": pi, "words": words,
                            "text": " ".join(w["word"] for w in words)})
            n_words += len(words)
        n_phrases += len(phrases)
        verses.append({"ref": f"{book} {chap}:{vnum}", "book": book,
                       "chapter": chap, "verse": vnum, "phrases": phrases})
    return {"parasha": parasha, "parasha_name": name, "aliya": aliya,
            "verses": verses, "counts": {"verses": len(verses),
                                         "phrases": n_phrases, "words": n_words}}


@app.get("/api/verse_tree")
def verse_tree(book: str, chapter: int, verse: int):
    """The cantillation parse tree for one verse, plus a plain-Hebrew
    explanation of each division -- what a learner is actually shown."""
    tg = get_taam_grammar()
    path = PROJECT_DIR / "Data_Prep" / "sefaria_cache" / f"{book}.{chapter}.json"
    if not path.exists():
        return {"error": f"no cached text for {book} {chapter}"}
    with open(path, encoding="utf-8") as f:
        verses = json.load(f)
    if verse < 1 or verse > len(verses):
        return {"error": f"{book} {chapter} has {len(verses)} verses"}
    tree, tokens = tg.parse_verse(verses[verse - 1])
    if tree is None:
        return {"error": "could not parse"}
    return {
        "ref": f"{book} {chapter}:{verse}",
        "words": [{"word": w, "taam": t, "taam_he": tg.TAAM_HE.get(t),
                   "role": ("מפסיק" if t in tg.DISJUNCTIVE_RANK
                            else "משרת" if t in tg.CONJUNCTIVE else None),
                   "rank": tg.DISJUNCTIVE_RANK.get(t),
                   "rank_he": tg.RANK_HE.get(tg.DISJUNCTIVE_RANK.get(t))}
                  for w, t in tokens],
        "tree": tree,
        "explanation": tg.describe(tree, tokens),
        "depth": tg.tree_depth(tree),
    }


@app.get("/api/verse_books")
def verse_books():
    """Which books/chapters have cached cantillated text available."""
    out = {}
    d = PROJECT_DIR / "Data_Prep" / "sefaria_cache"
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        book, ch = f.stem.rsplit(".", 1)
        out.setdefault(book, []).append(int(ch))
    return {b: sorted(v) for b, v in out.items()}


# ---------------------------------------------------- manual cut editor --
# webapp/static/cut.html. Ben13's own text is not a reliable transcript of
# its own audio (confirmed by review: several clips list words that are
# never sung), and a forced alignment handed too many words compresses
# silently rather than failing -- so every boundary after the phantom word
# is wrong, with a perfectly ordinary-looking confidence score. The only
# fix is a human correcting the transcript and the cuts by ear.
@app.get("/api/cut/clips")
def cut_clips():
    return get_cut_clips()


@app.get("/api/cut/taamim")
def cut_taamim():
    """Every ta'am class the project knows, for the per-word dropdown."""
    spec = importlib.util.spec_from_file_location(
        "clean_align_v2", PROJECT_DIR / "Data_Prep" / "clean_align_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sorted(set(mod.TAAM_MAP.values())) + ["Siluk"]


@app.get("/api/cut/audio/{key}")
def cut_audio(key: str):
    key = safe_clip_id(key)
    for row in get_cut_clips():
        if row["key"] == key:
            source = get_ben13_source_index().get(
                (row["source_type"], row["source_order"]))
            if source:
                p = PROJECT_DIR / source["local_path"]
                if p.exists():
                    return FileResponse(p, media_type="audio/mpeg")
    return {"error": "clip not found"}


@app.post("/api/cut/save")
def cut_save(payload: dict = Body(...)):
    key = safe_clip_id(payload.get("key", ""))
    if not key:
        return {"error": "missing key"}
    words = payload.get("words") or []
    for w in words:
        try:
            w["start"], w["end"] = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            return {"error": "each word needs numeric start/end"}
        if w["end"] <= w["start"]:
            return {"error": f"word {w.get('label','')!r} ends before it starts"}
    save_manual_cut(key, words, payload.get("note", ""))
    return {"ok": True, "saved": len(words)}


@app.post("/api/review/sephardi_name")
def review_sephardi_name(payload: dict = Body(...)):
    """One name per ta'am CLASS, not per clip -- Ben13's own scraped text is
    too inconsistent to derive this automatically (the same class shows up
    under different spoken names in different clips; confirmed by hand, see
    results/v3_comparison/ben13_segment_review.md), so this is a reviewer-
    entered label, editable from the review UI, shown next to every clip of
    that class."""
    taam = payload.get("taam", "")
    if not taam:
        return {"error": "missing taam"}
    save_sephardi_name(taam, payload.get("name", ""))
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
