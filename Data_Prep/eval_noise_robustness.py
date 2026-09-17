"""
What happens when the microphone is not a studio?
=================================================
Every word this project has ever trained on is clean: Ben13 is a produced
teaching product, recorded for publication. The product runs on a laptop or
phone microphone, in a room with a fan, a sibling and a street outside.
Nothing in training has represented that, and nothing has ever measured it.

That gap is measurable today, with no new recordings: take the held-out
parashot the model has never seen, raise a noise floor over them, and watch
where accuracy goes. Two models are compared -- the shipped one, and a twin
trained with --noise -- so the answer is not just "it degrades" but "and
here is whether the existing augmentation fixes it."

The noise model is deliberately the SAME one training uses (`noise_augment`
in 10_train_pipeline.py): on a mel-dB spectrogram, room noise is a floor
that lifts the quiet bins and leaves the loud ones alone. Measuring against
a different noise model than the one we can train against would answer a
question we cannot act on.

Two numbers are reported at each level, because they are different
questions:

  * classifier accuracy  -- can it still name the ta'am
  * contradiction rate   -- how often the verify head tells a learner who
                            read CORRECTLY that they were wrong. This is the
                            expensive error (see calibrate_verify_head.py),
                            and it is what actually breaks the product in a
                            noisy room.

Usage:
  python Data_Prep/eval_noise_robustness.py \
      --checkpoints results/A/deep_cnn/deep_cnn_best.pth \
      --labels shipped --data-dir prepared_data/parasha8_top21_torahhonest
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "results" / "noise_robustness"

# Fraction of each clip's own dynamic range that the noise floor is raised
# to. 0.0 is the clean corpus; training's --noise draws uniformly up to
# 20/60 = 0.33, so everything past that is beyond what augmentation has
# ever shown the model.
LEVELS = [0.0, 0.08, 0.17, 0.33, 0.50]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_noise_floor(X: np.ndarray, frac: float, rng) -> np.ndarray:
    """Deterministic version of the training-time noise_augment: lift every
    clip's floor to `frac` of its own dynamic range. Training randomises the
    level per sample; an evaluation must fix it, or the curve is an average
    over an unknown mixture."""
    if frac <= 0:
        return X
    out = X.astype(np.float32, copy=True)
    axes = tuple(range(1, out.ndim))
    lo = out.min(axis=axes, keepdims=True)
    hi = out.max(axis=axes, keepdims=True)
    floor = lo + (hi - lo) * frac
    noisy = floor + rng.standard_normal(out.shape).astype(np.float32) * 0.15
    return np.maximum(out, noisy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--verify-head", default=None,
                    help="if given, also measure the contradiction rate -- how "
                         "often a correctly-read word is called wrong")
    ap.add_argument("--verify-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=4000)
    args = ap.parse_args()
    assert len(args.checkpoints) == len(args.labels)

    tp = load_module("tp", "10_train_pipeline.py")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(0)

    test = np.load(Path(args.data_dir) / "test.npz")
    X, y = test["X"], test["y"]
    if len(X) > args.limit:
        keep = rng.choice(len(X), size=args.limit, replace=False)
        X, y = X[keep], y[keep]
    meta = json.loads((Path(args.data_dir) / "metadata.json").read_text())
    class_names = meta["class_names"] if "class_names" in meta else meta.get("classes")
    print(f"test set: {len(X):,} words, {len(set(y.tolist()))} classes "
          f"(held-out parashot — never trained on)\n")

    head = None
    if args.verify_head:
        bv = load_module("bv", "15_binary_verify_model.py")
        hck = torch.load(args.verify_head, map_location="cpu", weights_only=False)
        head = bv.VerifyHead(hck["emb_dim"], len(hck["class_names"]), hck["hidden"])
        head.load_state_dict(hck["model_state_dict"])
        head.eval().to(device)
        head_classes = list(hck["class_names"])

    L = []
    def say(s=""):
        L.append(s); print(s, flush=True)

    say("=" * 70)
    say("  Noise robustness — held-out parashot, studio audio degraded")
    say("=" * 70)
    say("\n  noise floor = fraction of each clip's own dynamic range.")
    say("  training's --noise draws uniformly up to 0.33, so 0.50 is past")
    say("  anything the augmentation has ever shown the model.\n")

    results = {}
    for ckpt_path, label in zip(args.checkpoints, args.labels):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = tp.MODEL_REGISTRY[ck["architecture"]](num_classes=len(ck["class_names"]))
        model.load_state_dict(ck["model_state_dict"])
        model.eval().to(device)
        names = list(ck["class_names"])

        say(f"  {label}")
        say(f"  {'floor':>7}{'accuracy':>11}{'balanced':>11}{'drop':>9}"
            + (f"{'contradicts a correct kid':>28}" if head else ""))
        say("  " + "-" * (38 + (28 if head else 0)))
        base = None
        rows = []
        for frac in LEVELS:
            Xn = add_noise_floor(X, frac, np.random.default_rng(1234))
            # Embedding is the input to the final Linear layer, captured by a
            # hook -- the same way webapp/server.py does it, so this measures
            # the production path rather than a lookalike of it.
            import torch.nn as nn
            last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
            captured = []
            handle = last_linear.register_forward_hook(
                lambda _m, inp, _out: captured.append(inp[0].detach().cpu()))
            preds = []
            with torch.no_grad():
                for i in range(0, len(Xn), 256):
                    chunk = Xn[i:i + 256]
                    batch = torch.from_numpy(chunk).float()
                    if batch.ndim == 3:            # (N, mel, frames) -> add channel
                        batch = batch.unsqueeze(1)
                    batch = batch.to(device)
                    preds.append(model(batch).argmax(1).cpu().numpy())
            handle.remove()
            embs = captured
            pred = np.concatenate(preds)
            acc = accuracy_score(y, pred)
            bal = balanced_accuracy_score(y, pred)
            base = acc if base is None else base

            contradiction = None
            if head is not None:
                emb = torch.cat(embs).to(device)
                onehot = torch.zeros(len(y), len(head_classes), device=device)
                for j, cls in enumerate(y):
                    name = names[cls]
                    if name in head_classes:
                        onehot[j, head_classes.index(name)] = 1.0
                with torch.no_grad():
                    prob = torch.sigmoid(head(emb, onehot)).squeeze(-1).cpu().numpy()
                contradiction = float((prob < args.verify_threshold).mean())

            rows.append({"floor": frac, "accuracy": acc, "balanced": bal,
                         "contradiction_rate": contradiction})
            extra = f"{contradiction*100:>27.1f}%" if contradiction is not None else ""
            say(f"  {frac:>7.2f}{acc:>11.3f}{bal:>11.3f}{acc-base:>+9.3f}{extra}")
        results[label] = rows
        say("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "noise_robustness.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {OUT}/noise_robustness.json")


if __name__ == "__main__":
    main()
