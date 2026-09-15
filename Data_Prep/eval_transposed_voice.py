"""
Can the model handle a voice it has never heard — a higher one?
=================================================================
THE DISAGREEMENT THIS SETTLES (2026-09-09)

Every voice in this corpus is an adult male ba'al kri'ah: R1, R2, R8, and
Ben13. The product's user is a bar-mitzvah-age learner, whose voice sits
roughly an octave higher. I argued this was a serious gap. The user's
position: *"אפשר ללמד אותו גם בלי לשמוע בפועל קול של ילד"* — you can teach
the model to handle it without ever hearing one.

That is a testable claim, so this tests it rather than arguing it.

METHOD
------
Take a trained model's own test set and transpose it upward on the mel axis
(a log-frequency axis, so a constant bin shift ≈ a constant ratio in Hz).
+6 to +12 bins spans roughly the register gap between an adult man and a
child. Then measure accuracy at each shift, for two models:

  * one trained normally
  * one trained with --pitch-shift augmentation

If the augmented model holds its accuracy on transposed audio while the
plain one collapses, the user is right: the register gap is fixable without
a single child recording. If BOTH collapse, real recordings are required.
If NEITHER drops, the model was already register-invariant and the whole
concern was misplaced.

Reports all three outcomes plainly — the point is the measurement, not
winning the argument.

Usage:
  python Data_Prep/eval_transposed_voice.py \
      --checkpoints results/A/deep_cnn/deep_cnn_best.pth results/B/deep_cnn/deep_cnn_best.pth \
      --labels plain pitch_aug --data-dir prepared_data/random_top21_prodA
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
SHIFTS = [0, 3, 6, 9, 12]


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "tp", PROJECT_DIR / "10_train_pipeline.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    return tp


def transpose(X: np.ndarray, bins: int) -> np.ndarray:
    """Shift up the mel axis by `bins`, filling vacated rows with the floor."""
    if bins == 0:
        return X
    out = np.roll(X, shift=bins, axis=-2)
    floor = X.min(axis=(-2, -1), keepdims=True)
    out[..., :bins, :] = np.broadcast_to(floor, out[..., :bins, :].shape)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/transposed_voice/report.txt")
    args = ap.parse_args()

    tp = load_pipeline()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    d = np.load(Path(args.data_dir) / "test.npz")
    X, y = d["X"], d["y"]
    if X.ndim == 3:
        X = X[:, None, :, :]
    print(f"test set: {X.shape}")

    L = []
    def say(s=""):
        L.append(s); print(s, flush=True)

    say("=" * 66)
    say("  Does the model survive a higher voice it never trained on?")
    say("=" * 66)
    say(f"\n  test set: {len(X):,} words from {args.data_dir}")
    say("  a mel-bin shift is ~a transposition; +6..+12 spans the adult-male")
    say("  to child register gap\n")
    header = f"  {'model':<14}" + "".join(f"{'+' + str(s) + ' bins':>12}" for s in SHIFTS)
    say(header)
    say("  " + "-" * (14 + 12 * len(SHIFTS)))

    results = {}
    for ckpt_path, label in zip(args.checkpoints, args.labels):
        if not Path(ckpt_path).exists():
            say(f"  {label:<14}(checkpoint missing: {ckpt_path})")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = tp.MODEL_REGISTRY[ck["architecture"]](num_classes=len(ck["class_names"]))
        model.load_state_dict(ck["model_state_dict"])
        model.eval().to(device)

        row, accs = f"  {label:<14}", {}
        for s in SHIFTS:
            Xs = transpose(X, s).astype(np.float32)
            preds = []
            with torch.no_grad():
                for i in range(0, len(Xs), 256):
                    xb = torch.from_numpy(Xs[i:i + 256]).to(device)
                    preds.append(model(xb).argmax(1).cpu().numpy())
            p = np.concatenate(preds)
            a = accuracy_score(y, p)
            accs[s] = {"accuracy": float(a),
                       "balanced": float(balanced_accuracy_score(y, p))}
            row += f"{a*100:>11.1f}%"
        say(row)
        results[label] = accs

    if len(results) >= 1:
        say("")
        for label, accs in results.items():
            drop = accs[0]["accuracy"] - accs[max(SHIFTS)]["accuracy"]
            say(f"  {label}: {drop*100:+.1f} points lost from 0 to +{max(SHIFTS)} bins")

    say("\n" + "=" * 66)
    out = PROJECT_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.parent / "report.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
