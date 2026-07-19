"""
12_gradcam.py — Grad-CAM for the v2 pipeline checkpoints
========================================================
Shows WHERE on the mel-spectrogram the model listens when it recognizes each
ta'am (the "not a black box" evidence). Hooks the last Conv2d layer, so it
works with any convolutional architecture in the registry (cnn, deep_cnn,
resnet, crnn, crnn_lstm, resnet18_tl, transformer frontend). Pure-sequence
models (mlp, bilstm) have no conv layer and are rejected.

Usage:
  python 12_gradcam.py --checkpoint results/random_top11/cnn/cnn_best.pth \
                       --data-dir prepared_data/random_top11
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent
OUT_DIR = PROJECT_DIR / "analysis" / "gradcam"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_pipeline", PROJECT_DIR / "10_train_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gradcam(model, conv_layer, x, target_class):
    """Gradient-weighted class activation map for one sample (1,1,128,87)."""
    acts, grads = [], []
    h1 = conv_layer.register_forward_hook(lambda _m, _i, o: acts.append(o))
    h2 = conv_layer.register_full_backward_hook(lambda _m, _gi, go: grads.append(go[0]))

    model.zero_grad()
    logits = model(x)
    logits[0, target_class].backward()
    h1.remove()
    h2.remove()

    act, grad = acts[0][0], grads[0][0]              # (C, h, w)
    weights = grad.mean(dim=(1, 2))                  # GAP over spatial dims
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    # upsample to input resolution
    cam_t = torch.from_numpy(cam)[None, None]
    cam_up = torch.nn.functional.interpolate(
        cam_t, size=(128, 87), mode='bilinear', align_corners=False)[0, 0].numpy()
    confidence = torch.softmax(logits, dim=1)[0, target_class].item()
    return cam_up, confidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True,
                    help='prepared_data/<scenario> with test.npz + metadata.json')
    ap.add_argument('--per-class', type=int, default=1)
    args = ap.parse_args()

    train_mod = load_train_module()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    class_names = ckpt['class_names']
    model = train_mod.MODEL_REGISTRY[ckpt['architecture']](num_classes=len(class_names))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    conv_layer = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layer = m  # keep the LAST one
    if conv_layer is None:
        raise SystemExit(f"{ckpt['architecture']} has no Conv2d layer — Grad-CAM "
                         "needs a convolutional model (try cnn/resnet/crnn).")

    test = np.load(args.data_dir / 'test.npz')
    X, y = test['X'], test['y']

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 256):
            preds.append(model(torch.from_numpy(X[i:i + 256])).argmax(dim=1))
        preds = torch.cat(preds).numpy()

    for cls_idx, cls_name in enumerate(class_names):
        correct = np.where((y == cls_idx) & (preds == cls_idx))[0]
        if len(correct) == 0:
            print(f"  {cls_name}: no correctly-classified sample, skipping")
            continue
        for k, idx in enumerate(correct[:args.per_class]):
            x = torch.from_numpy(X[idx:idx + 1]).requires_grad_(True)
            cam, conf = gradcam(model, conv_layer, x, cls_idx)
            spec = X[idx, 0]

            fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
            axes[0].imshow(spec, aspect='auto', origin='lower', cmap='magma')
            axes[0].set_title(f'Mel-spectrogram — {cls_name}')
            axes[0].set_xlabel('Time frame')
            axes[0].set_ylabel('Mel bin')
            axes[1].imshow(cam, aspect='auto', origin='lower', cmap='inferno')
            axes[1].set_title('Grad-CAM — where the model listens')
            axes[1].set_xlabel('Time frame')
            axes[2].imshow(spec, aspect='auto', origin='lower', cmap='gray')
            axes[2].imshow(cam, aspect='auto', origin='lower', cmap='inferno', alpha=0.5)
            axes[2].set_title(f'Overlay (confidence {conf:.0%})')
            axes[2].set_xlabel('Time frame')
            fig.suptitle(f"{ckpt['architecture']} — {cls_name}", fontweight='bold')
            plt.tight_layout()
            out = OUT_DIR / f"gradcam_{cls_name}{'' if k == 0 else f'_{k}'}.png"
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  {cls_name}: saved {out.name} (confidence {conf:.0%})")

    print(f"\nGrad-CAM figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
