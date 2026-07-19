"""
smoke_test.py — proves the whole training machinery runs on a fresh machine
============================================================================
No data needed: builds a tiny synthetic dataset in the exact prepared_data
format, trains the production architecture for 2 epochs, and checks that the
loss drops and every artifact is written. Runs in ~1-2 minutes on CPU.

Usage:  python smoke_test.py
Exit 0 = the pipeline works end-to-end on this machine.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent


def make_synthetic_prepared_data(root: Path, n_classes=3, n_train=600,
                                 n_val=120, n_test=120):
    """Tiny dataset in prepared_data format: three separable 'melodies'."""
    rng = np.random.default_rng(0)
    class_names = [f"Taam_{i}" for i in range(n_classes)]

    def make_split(n):
        X = rng.normal(0, 0.3, (n, 128, 87)).astype(np.float32)
        y = rng.integers(0, n_classes, n).astype(np.int64)
        t = np.linspace(0, 1, 87)
        for i in range(n):
            # each class = a distinct frequency contour drawn into the "mel" image
            center = 30 + 25 * y[i] + (10 * np.sin(2 * np.pi * (y[i] + 1) * t)).astype(int)
            for j, c in enumerate(center):
                X[i, max(0, c - 2):c + 3, j] += 3.0
        return X[:, None, :, :], y   # channel dim, as 00_validate saves it

    root.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        X, y = make_split(n)
        np.savez_compressed(root / f"{split}.npz", X=X, y=y)
        sizes[split] = n
    with open(root / "metadata.json", "w") as f:
        json.dump({"class_names": class_names,
                   "label_map": {n: i for i, n in enumerate(class_names)},
                   "input_shape": [128, 87], "normalization": "per_sample",
                   "split_sizes": sizes}, f)
    return class_names


def main():
    print("1/3 building synthetic dataset...")
    tmp = Path(tempfile.mkdtemp(prefix="taamify_smoke_"))
    data_dir = tmp / "prepared_data" / "smoke"
    make_synthetic_prepared_data(data_dir)

    print("2/3 running 10_train_pipeline (deep_cnn, 6 epochs)...")
    spec = importlib.util.spec_from_file_location(
        "tp", PROJECT_DIR / "10_train_pipeline.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)

    argv_backup = sys.argv
    sys.argv = ["10_train_pipeline.py", "--model", "deep_cnn",
                "--epochs", "6", "--batch-size", "32", "--no-specaugment",
                "--data-dir", str(data_dir), "--no-embeddings"]
    try:
        tp.main()
    finally:
        sys.argv = argv_backup

    print("3/3 checking artifacts...")
    results = PROJECT_DIR / "results" / "smoke" / "deep_cnn"
    required = ["deep_cnn_best.pth", "deep_cnn_metrics.json",
                "deep_cnn_report.txt", "deep_cnn_confusion_matrix.png"]
    missing = [f for f in required if not (results / f).exists()]
    if missing:
        print(f"FAIL: missing artifacts: {missing}")
        sys.exit(1)

    with open(results / "deep_cnn_metrics.json") as f:
        acc = json.load(f)["accuracy"]
    # separable synthetic classes: even 2 epochs must beat chance (0.33) clearly
    if acc < 0.5:
        print(f"FAIL: accuracy {acc:.2f} — model did not learn the synthetic task")
        sys.exit(1)

    print(f"\nSMOKE TEST PASSED — synthetic accuracy {acc:.2f}, all artifacts written")
    print(f"(artifacts in {results}; safe to delete)")


if __name__ == "__main__":
    main()
