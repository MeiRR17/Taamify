#!/bin/bash
# ============================================================================
# Job queue runner — keeps the machine full, refills the instant a slot frees.
#
#   jobs/queue/NNN_name[.gpu].job  pending   (.gpu -> GPU pool, else CPU pool)
#   jobs/running/  jobs/done/  jobs/failed/  jobs/logs/name.log
#
# Two pools, because they are two different resources. The GPU is ONE device:
# a dozen concurrent MPS trainings just timeshare it, and (measured 2026-09-09)
# sit at ~4% CPU each while half the CPU idles. Real throughput comes from a
# few GPU jobs plus CPU work that never wants the GPU.
#
# DEPENDENCIES — the fix for the failure that cost hours twice. Jobs used to
# wait for their input with `while [ ! -f X ]; do sleep 60; done` INSIDE the
# job, which holds a slot while doing nothing, and deadlocks outright when the
# job that produces X is still queued behind it. Instead declare inputs:
#
#     # NEEDS: results/foo/deep_cnn/deep_cnn_best.pth
#
# The runner skips a job whose NEEDS are unmet, leaves it in the queue, and
# moves on to the next. Nothing blocks, nothing deadlocks.
#
# RECONCILE — restarting the runner (to retune slots) kills the tracking
# subshells while their training processes survive as orphans, so a .job file
# sat in running/ forever, the queue believed it was full, and the machine
# idled ~13 hours on 2026-09-09. Every poll, a running job whose recorded pid
# is gone is sorted by what its log actually says.
#
# MEMORY GUARD — this project has lost hours to swap-thrash death more than
# once. A new job starts only above MIN_FREE_GB, so a full queue degrades into
# fewer slots instead of a crash that loses everything already running.
#
#   GPU_SLOTS=6 CPU_SLOTS=4 MIN_FREE_GB=6 ./run_queue.sh &
# ============================================================================
set -u
cd "$(dirname "$0")"
GPU_SLOTS=${GPU_SLOTS:-6}
CPU_SLOTS=${CPU_SLOTS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-6}
POLL=${POLL:-15}
Q=jobs/queue; R=jobs/running; D=jobs/done; F=jobs/failed; L=jobs/logs
mkdir -p $Q $R $D $F $L
LOG=jobs/QUEUE.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

free_gb(){
  vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3}
    END{gsub(/\./,"",f); gsub(/\./,"",i); printf "%d", (f+i)*16384/1073741824}'
}
n_running(){ ls $R/*.job 2>/dev/null | wc -l | tr -d ' '; }
n_gpu(){ ls $R/*.gpu.job 2>/dev/null | wc -l | tr -d ' '; }
n_cpu(){ echo $(( $(n_running) - $(n_gpu) )); }

needs_met(){   # $1 = job file; every "# NEEDS: <path>" line must exist
  local miss=0 path
  while read -r path; do
    [ -z "$path" ] && continue
    [ -e "$path" ] || { miss=1; break; }
  done < <(grep -E '^# *NEEDS:' "$1" 2>/dev/null | sed -E 's/^# *NEEDS: *//')
  [ "$miss" = 0 ]
}

pool_of(){ case "$1" in *.gpu.job) echo gpu;; *) echo cpu;; esac; }

reconcile(){
  local any=0 name pid
  for j in $R/*.job; do
    [ -e "$j" ] || continue
    name=$(basename "$j" .job)
    pid=$(cat "$R/$name.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then continue; fi
    any=1
    if grep -qE "ALL EXPERIMENTS COMPLETE|Results saved to|=== .* complete|^saved:" "$L/$name.log" 2>/dev/null; then
      mv "$j" "$D/$name.job" 2>/dev/null; rm -f "$R/$name.pid"
      say "ADOPT  $name — finished untracked, marked done"
    else
      mv "$j" "$Q/$name.job" 2>/dev/null; rm -f "$R/$name.pid"
      say "ADOPT  $name — tracker gone with no result, requeued"
    fi
  done
  [ "$any" = 1 ] && say "reconciled orphaned slots"
  return 0
}

launch(){
  local job=$1 mem=$2 name
  name=$(basename "$job" .job)
  mv "$job" "$R/$name.job" 2>/dev/null || return 1
  say "START  $name  ($(n_gpu)gpu+$(n_cpu)cpu of ${GPU_SLOTS}+${CPU_SLOTS}, ${mem}GB free)"
  (
    start=$(date +%s)
    # Track the JOB's own pid, not the subshell's. $BASHPID does not exist in
    # bash 3.2 (what macOS ships) and under `set -u` reading it killed every
    # subshell instantly, before it could launch anything -- the queue logged
    # START for six jobs and ran none of them. $$ is no good either: inside a
    # subshell it still reports the parent. Backgrounding the job and taking
    # $! is correct on 3.2 and gives reconcile a pid that means something.
    PYTHONUNBUFFERED=1 bash "$R/$name.job" > "$L/$name.log" 2>&1 &
    jpid=$!
    echo "$jpid" > "$R/$name.pid"
    if wait "$jpid"; then
      mv "$R/$name.job" "$D/$name.job" 2>/dev/null
      say "OK     $name ($(( ($(date +%s)-start)/60 ))m)"
    else
      mv "$R/$name.job" "$F/$name.job" 2>/dev/null
      say "FAIL   $name ($(( ($(date +%s)-start)/60 ))m) — jobs/logs/$name.log"
    fi
    rm -f "$R/$name.pid"
  ) &
}

reconcile
say "===== queue runner up: ${GPU_SLOTS} GPU + ${CPU_SLOTS} CPU slots, min ${MIN_FREE_GB}GB free ====="
idle=0
while true; do
  reconcile
  pend=$(ls $Q/*.job 2>/dev/null | wc -l | tr -d ' ')
  if [ "$(n_running)" -eq 0 ] && [ "$pend" -eq 0 ]; then
    idle=$((idle+1)); [ "$idle" -gt 40 ] && { say "queue empty — exiting"; break; }
  else idle=0; fi

  for job in $(ls $Q/*.job 2>/dev/null); do
    pool=$(pool_of "$job")
    if [ "$pool" = gpu ]; then [ "$(n_gpu)" -ge "$GPU_SLOTS" ] && continue
    else [ "$(n_cpu)" -ge "$CPU_SLOTS" ] && continue; fi
    needs_met "$job" || continue          # inputs not ready — leave it queued
    mem=$(free_gb)
    if [ "$mem" -lt "$MIN_FREE_GB" ]; then
      say "HOLD   ${mem}GB free < ${MIN_FREE_GB}GB"; break
    fi
    launch "$job" "$mem"
    sleep 3
  done
  sleep $POLL
done
