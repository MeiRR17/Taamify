"""
11_run_loro.py — Leave-One-Reader-Out evaluation orchestrator
=============================================================
For each holdout reader: prepares a LORO split (00_validate_and_prepare_data.py)
and trains/evaluates the chosen model(s) (10_train_pipeline.py), then aggregates
per-fold test metrics into results/loro_summary_<model>.md (mean ± std).

The headline experiment of the project: does the model generalize to a voice it
has never heard? Each fold answers it for one reader.

Usage:
  python 11_run_loro.py --model cnn --holdouts R2 R3 R4 R5 --num-classes 11
  python 11_run_loro.py --model resnet18_tl --epochs 40
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent
PYTHON = sys.executable

METRIC_KEYS = ['accuracy', 'balanced_accuracy', 'mcc', 'f1_macro', 'auc_macro']


def run(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='cnn',
                    help="model name(s), comma-separated, or 'all'. Multiple models "
                         "are trained on the same fold (built once).")
    ap.add_argument('--holdouts', nargs='+', default=['R2', 'R3', 'R4', 'R5'],
                    help='readers to hold out, one fold each (R1 is usually train-only: '
                         'it is the largest reader)')
    ap.add_argument('--readers', nargs='+', default=['R1', 'R2', 'R3', 'R4', 'R5'])
    ap.add_argument('--num-classes', type=int, default=11)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--skip-prep', action='store_true',
                    help='reuse existing prepared_data/loro_* folders')
    ap.add_argument('--keep-folds', action='store_true',
                    help='do not delete each fold prepared_data after training '
                         '(default: delete, to save disk on Google Drive)')
    args = ap.parse_args()

    fold_metrics = {}   # holdout -> model -> metrics dict

    tmp_root = Path("/tmp/taamify_loro")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for holdout in args.holdouts:
        scenario = f"loro_{holdout}_top{args.num_classes}"
        data_dir = PROJECT_DIR / 'prepared_data' / scenario

        if not args.skip_prep or not data_dir.exists():
            run([PYTHON, '00_validate_and_prepare_data.py',
                 '--readers', *args.readers,
                 '--num-classes', str(args.num_classes),
                 '--split', 'loro', '--holdout', holdout])

        # Train from a LOCAL copy: reading fold npz straight from the Drive-synced
        # folder races with sync and intermittently kills the child (seen in
        # practice). Results still write back to the Drive results/ tree.
        local_dir = tmp_root / scenario
        shutil.rmtree(local_dir, ignore_errors=True)
        shutil.copytree(data_dir, local_dir)

        for model_name in args.model.split(','):
            run([PYTHON, '10_train_pipeline.py',
                 '--data-dir', str(local_dir),
                 '--model', model_name.strip(),
                 '--epochs', str(args.epochs),
                 '--batch-size', str(args.batch_size)])

        # Collect metrics — pipeline names the results dir after the data-dir,
        # so under /tmp the scenario folder is the local_dir's name (== scenario)
        results_dir = PROJECT_DIR / 'results' / scenario
        fold_metrics[holdout] = {}
        for metrics_file in results_dir.glob('*/*_metrics.json'):
            model_name = metrics_file.parent.name
            with open(metrics_file) as f:
                fold_metrics[holdout][model_name] = json.load(f)

        # Free disk before the next fold (Drive volume is tight)
        shutil.rmtree(local_dir, ignore_errors=True)
        if not args.keep_folds:
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"  cleaned {data_dir}")

    # Aggregate across folds per model
    model_names = sorted({m for fold in fold_metrics.values() for m in fold})
    lines = [f"# LORO Summary — top-{args.num_classes} classes",
             "",
             f"Folds (held-out reader): {', '.join(args.holdouts)}",
             ""]
    for model_name in model_names:
        lines.append(f"\n## {model_name}\n")
        header = "| Metric | " + " | ".join(args.holdouts) + " | Mean ± Std |"
        lines.append(header)
        lines.append("|" + "---|" * (len(args.holdouts) + 2))
        for key in METRIC_KEYS:
            vals = [fold_metrics.get(h, {}).get(model_name, {}).get(key) for h in args.holdouts]
            cells = [f"{v:.4f}" if v is not None else "—" for v in vals]
            present = [v for v in vals if v is not None]
            agg = f"{np.mean(present):.4f} ± {np.std(present):.4f}" if present else "—"
            lines.append(f"| {key} | " + " | ".join(cells) + f" | {agg} |")

    out_path = PROJECT_DIR / 'results' / f"loro_summary_{args.model}_top{args.num_classes}.md"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLORO summary -> {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
