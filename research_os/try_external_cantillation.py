"""Local inference smoke test for a published cantillation checkpoint.

Downloads weights from Hugging Face; never uploads audio or trains a model.
The default is the published Jerusalem-specific Whisper-small checkpoint.
Output is local-only and is NOT a generalization/learner-error benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import librosa
import torch
import truststore
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from webapp.channel_calibration import sha256_file

ROOT = Path(__file__).resolve().parent.parent
MODEL = "cantillation/Teamim-small_WeightDecay-0.05_Augmented_New-Data_nusach-yerushalmi_date-24-07-2024"
REVISION = "53028d404fd534f414a9fa6570b00da57c3288db"
VARIANTS = {
    "yerushalmi-small": (MODEL, REVISION),
    "tiny-smoke": ("cantillation/Teamim-tiny_WeightDecay-0.005_Augmented_WithSRT_date-15-05-2025",
                   "0984378f6f18418ab8f050afd2613f7ed1b67003"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="yerushalmi-small")
    args = ap.parse_args()
    model_id, revision = VARIANTS[args.variant]
    if args.out.exists():
        ap.error("output already exists; preserve previous evidence")
    truststore.inject_into_ssl()
    torch.set_num_threads(1)
    waveform, sr = librosa.load(args.audio, sr=16000, mono=True)
    if not 0.2 <= len(waveform) / sr <= 30:
        ap.error("use a real 0.2..30 second phrase, without splicing")
    info = HfApi().model_info(model_id, revision=revision)
    license_id = (info.card_data or {}).get("license")
    if license_id != "apache-2.0":
        raise RuntimeError("published checkpoint license metadata changed; review required")
    start = time.monotonic()
    local = snapshot_download(model_id, revision=revision,
                              local_dir=ROOT / "models/external_cantillation" / args.variant,
                              allow_patterns=["*.json", "*.txt", "*.safetensors"])
    download_seconds = time.monotonic() - start
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(local, local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval().to(device)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt",
                       return_attention_mask=True).to(device)
    if device == "mps":
        torch.mps.synchronize()
    load_seconds = time.monotonic() - start
    start = time.monotonic()
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                language="he", task="transcribe")
    if device == "mps":
        torch.mps.synchronize()
    inference_seconds = time.monotonic() - start
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    report = {"status": "inference_smoke_only", "model_id": model_id,
              "variant": args.variant, "revision": revision, "declared_license": license_id,
              "weights_sha256": sha256_file(Path(local) / "model.safetensors"),
              "audio_sha256": sha256_file(args.audio),
              "audio_seconds": round(len(waveform) / sr, 3),
              "sample_rate": sr, "device": device, "torch_version": torch.__version__,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runner_sha256": sha256_file(__file__),
              "download_seconds": round(download_seconds, 3),
              "load_seconds": round(load_seconds, 3),
              "inference_seconds": round(inference_seconds, 3),
              "transcription": text,
              "limitations": ["no accuracy or learner-error claim",
                              "possible overlap with Ben13 training audio",
                              "no expected verse or reference text supplied to decoder"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("transcription", "audio_sha256", "weights_sha256")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
