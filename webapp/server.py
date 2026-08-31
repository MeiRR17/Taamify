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
import json
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from fastapi import Body, FastAPI, File, Form, UploadFile
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
VERIFY_HEAD_PATH = PROJECT_DIR / "results" / "binary_verify" / "production_verify_head.pth"
ATTEMPTS_DIR = PROJECT_DIR / "app_assets" / "collected_attempts"
ATTEMPTS_LOG = ATTEMPTS_DIR / "attempts_log.jsonl"
BEN13_SEGMENTS_DIR = PROJECT_DIR / "data" / "ben13_segments"
BEN13_SEGMENTS_MANIFEST = BEN13_SEGMENTS_DIR / "manifest.json"
BEN13_SOURCE_DIR = PROJECT_DIR / "data" / "ben13"
REVIEW_NOTES_PATH = BEN13_SEGMENTS_DIR / "review_notes.json"

# audio constants — must match extract_features_v2.py
SR = 22050
N_MELS = 128
FMAX = 8000
MAX_FRAMES = 87
PAD_DB = -80.0

app = FastAPI(title="Taamify")

# ---------------------------------------------------------------- models --
_state = {}


def get_model():
    if "model" not in _state:
        spec = importlib.util.spec_from_file_location(
            "tp", PROJECT_DIR / "10_train_pipeline.py")
        tp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tp)
        ckpt_path = PROJECT_DIR / "results/random_top11/deep_cnn/deep_cnn_best.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = tp.MODEL_REGISTRY[ckpt["architecture"]](
            num_classes=len(ckpt["class_names"]))
        model.load_state_dict(ckpt["model_state_dict"])
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.eval().to(device)
        _state.update(model=model, device=device,
                      class_names=ckpt["class_names"],
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
        notes[clip_id] = {"note": note, "flag": flag}
    else:
        notes.pop(clip_id, None)
    REVIEW_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=1)


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
    items = []
    for e in segments:
        clip_id = Path(e["local_path"]).stem
        source = source_idx.get((e["source_type"], e["source_order"]))
        n = notes.get(clip_id, {})
        items.append({
            "clip_id": clip_id,
            "taam": e["taam"],
            "source_type": e["source_type"],
            "source_order": e["source_order"],
            "ctc_score": e["ctc_score"],
            "duration": e["duration"],
            "low_confidence": e["low_confidence"],
            "source_text": source["text"] if source else None,
            "source_text_sfaradi": source["textSfaradi"] if source else None,
            "note": n.get("note", ""),
            "flag": n.get("flag", ""),
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


def classify(spec, top_k=3):
    st = get_model()
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
    return prob >= 0.5, prob


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
            "curve": rec["f0_semitones"]})
    return by_taam


@app.get("/api/teacher/audio/{clip_id}")
def teacher_audio(clip_id: str):
    safe = "".join(c for c in clip_id if c.isalnum() or c == "_")
    return FileResponse(TEACHER_DIR / f"{safe}.wav", media_type="audio/wav")


@app.post("/api/practice/{clip_id}")
async def practice(clip_id: str, audio: UploadFile = File(...)):
    rec = get_teacher().get(clip_id)
    if rec is None:
        return {"error": "unknown clip"}
    y = load_wav(await audio.read())
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
    return {"is_match": is_match, "confidence": round(confidence, 3),
            "expected_taam": expected_taam}


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    y = load_wav(await audio.read())
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    segments, _ = get_whisper().transcribe(
        y16, language="he", word_timestamps=True, beam_size=5)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            spec = word_spectrogram(y, w.start, w.end)
            if spec is None:
                continue
            top3, _ = classify(spec)
            words.append({"word": w.word.strip(),
                          "start": round(w.start, 2), "end": round(w.end, 2),
                          "predictions": [{"taam": t, "prob": round(p, 3)}
                                          for t, p in top3]})
    return {"words": words}


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


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
