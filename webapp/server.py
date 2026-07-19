"""
Taamify web application — FastAPI backend
=========================================
Serves the single-page frontend (webapp/static/) and three API groups:

  GET  /api/info                     model + dataset facts for the header
  GET  /api/teacher                  practice catalogue (per-ta'am word clips)
  GET  /api/teacher/audio/{clip_id}  the teacher's audio clip
  POST /api/practice/{clip_id}       student WAV -> hybrid score + curves
  POST /api/analyze                  full recording WAV -> per-word ta'am list

The browser records raw PCM and uploads standard WAV, so no ffmpeg is needed.
Feature extraction matches the training pipeline exactly.

Run:  uvicorn webapp.server:app --port 8077        (from the project root)
"""

import importlib.util
import io
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
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


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
