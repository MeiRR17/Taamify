#!/bin/bash
# Waits for the training chain to exit, then writes the summary — so a
# finished run means a table is already waiting, not logs to grep.
cd "$(dirname "$0")"
while pgrep -f "run_ben13_training.sh" > /dev/null; do sleep 120; done
sleep 10
venv/bin/python collect_ben13_results.py \
  --json results/ben13_train/summary.json >> results/ben13_train/PROGRESS.log 2>&1
echo "[$(date '+%m-%d %H:%M')] SUMMARY written" >> results/ben13_train/PROGRESS.log
