#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "run_phrase_realign.sh" > /dev/null; do sleep 120; done
sleep 10
venv/bin/python collect_phrase_results.py >> results/phrase_realign/PROGRESS.log 2>&1
echo "[$(date '+%m-%d %H:%M')] SUMMARY written" >> results/phrase_realign/PROGRESS.log
