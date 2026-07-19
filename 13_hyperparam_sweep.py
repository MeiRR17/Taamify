"""
13_hyperparam_sweep.py — hyperparameter study on the production candidate
=========================================================================
Course requirement: "בדיקת מטה-פרמטרים שונים עד להגעה לפתרון משביע רצון".

Two stages, all runs short (--epochs 12) and selected on VALIDATION metrics
(test is reported only for the final chosen config, trained fully elsewhere):

  Stage 1 — grid:      lr x batch-size          (6 runs)
  Stage 2 — ablations: no SpecAugment / no L2 / patience effect (3 runs)

Each run gets its own results dir via 10_train_pipeline's --tag.
Summary table -> results/hyperparam_sweep.md

Usage:
  python 13_hyperparam_sweep.py --data-dir /tmp/prepared_data_random_top11
  python 13_hyperparam_sweep.py --model deep_cnn --epochs 12
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_config(args, tag, extra):
    # resume: a config whose metrics already exist is not re-trained
    metrics_file = (PROJECT_DIR / 'results' / f"{args.data_dir.name}_{tag}"
                    / args.model / f"{args.model}_metrics.json")
    if metrics_file.exists():
        print(f"  SKIP {tag} (metrics exist)", flush=True)
        with open(metrics_file) as f:
            return json.load(f)
    cmd = [PYTHON, '-u', '10_train_pipeline.py',
           '--data-dir', str(args.data_dir),
           '--model', args.model,
           '--epochs', str(args.epochs),
           '--patience', str(args.patience),
           '--no-embeddings',
           '--tag', tag] + extra
    print(f"\n{'='*70}\n  SWEEP RUN: {tag}  {' '.join(extra)}\n{'='*70}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"  RUN {tag} FAILED — continuing", flush=True)
        return None
    metrics_file = (PROJECT_DIR / 'results' / f"{args.data_dir.name}_{tag}"
                    / args.model / f"{args.model}_metrics.json")
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path,
                    default=Path('/tmp/prepared_data_random_top11'))
    ap.add_argument('--model', default='deep_cnn')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--patience', type=int, default=6)
    args = ap.parse_args()

    configs = []
    # Stage 1: learning rate x batch size
    for lr in ['1e-3', '5e-4', '1e-4']:
        for bs in ['32', '128']:
            configs.append((f"hp_lr{lr}_bs{bs}",
                            ['--lr', lr, '--batch-size', bs]))
    # Stage 2: ablations around the default config (lr 5e-4, bs 64)
    configs += [
        ("hp_default", ['--lr', '5e-4', '--batch-size', '64']),
        ("hp_no_specaug", ['--lr', '5e-4', '--batch-size', '64', '--no-specaugment']),
        ("hp_no_l2", ['--lr', '5e-4', '--batch-size', '64', '--weight-decay', '0']),
    ]

    rows = []
    for tag, extra in configs:
        m = run_config(args, tag, extra)
        if m:
            rows.append((tag, extra, m))

    rows.sort(key=lambda r: -r[2].get('val_balanced_accuracy', 0))
    lines = [f"# Hyperparameter Sweep — {args.model}, {args.epochs} epochs each",
             "", "Selected on **validation** balanced accuracy (test untouched).", "",
             "| Config | Val Acc | Val BalAcc | Val MCC | Best epoch |",
             "|---|---|---|---|---|"]
    for tag, extra, m in rows:
        lines.append(f"| {tag.replace('hp_','')} "
                     f"| {m.get('val_accuracy', 0)*100:.1f}% "
                     f"| {m.get('val_balanced_accuracy', 0)*100:.1f}% "
                     f"| {m.get('val_mcc', 0):.3f} "
                     f"| {m.get('best_epoch', '—')} |")
    out = PROJECT_DIR / 'results' / 'hyperparam_sweep.md'
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"\nSweep summary -> {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
