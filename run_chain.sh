#!/bin/bash
# Waits for part 1 to finish, then runs part 2. Keeps the whole ~58h queue
# unattended without editing a script that bash is still reading.
cd "$(dirname "$0")"
while pgrep -f "run_48h_sweep.sh" > /dev/null; do sleep 120; done
echo "[chain] part 1 finished $(date) — starting part 2" >> results/sweep_48h/PROGRESS.log
exec ./run_sweep_part2.sh
