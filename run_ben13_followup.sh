#!/bin/bash
# ============================================================================
# Ben13 follow-up: the two things the first programme left unanswered
# ============================================================================
# 1. BLOCK 2 (all classes) never ran -- --all-classes died on classes with a
#    single member. Fixed via --min-per-class; re-run it.
#
# 2. The headline negative ("adding Ben13 makes LORO worse, -2.9 points") is
#    confounded. --max-train stratifies by class, not by reader, and BEN13R is
#    53.5% of the available pool -- so ~21,000 of 40,000 training words came
#    from one voice. That tests "does one voice dominating half the training
#    hurt?", not "is this corpus good?". --balance-readers caps every reader
#    to the smallest one, which separates the two.
#
# Same conventions as before: done/ markers, per-step logs, hard gate on the
# features actually existing, no glob inlined into an argument list.
# ============================================================================
set -u
cd "$(dirname "$0")"

PY=venv/bin/python
FEAT=$PWD/features_v3_word
OUT=results/ben13_train
mkdir -p $OUT/done $OUT/logs
LOG=$OUT/PROGRESS.log
started=$(date +%s)

say(){ echo "[$(date '+%m-%d %H:%M')] [F+$(( ($(date +%s)-started)/60 ))m] $*" | tee -a $LOG; }
stop(){ [ -f $OUT/STOP ] && { say "STOP — exiting"; exit 0; }; }
run(){
  local name=$1; shift
  stop
  [ -f "$OUT/done/$name" ] && { say "SKIP $name"; return 0; }
  say "START $name"; local t0=$(date +%s)
  if "$@" > "$OUT/logs/$name.log" 2>&1; then
    say "OK    $name ($(( ($(date +%s)-t0)/60 ))m)"; touch "$OUT/done/$name"
  else
    say "FAIL  $name ($(( ($(date +%s)-t0)/60 ))m) — see logs/$name.log"
  fi
}

[ -f features_v3_word/BEN13R_X.npy ] || { say "ABORT: BEN13R features missing"; exit 1; }
say "===== FOLLOW-UP: block 2 re-run + reader-balanced comparison ====="

# ---------------------------------------------------------------------------
# PART A — block 2, all classes, now that the single-member classes are
# dropped rather than crashing the stratified split.
# ---------------------------------------------------------------------------
say "--- PART A: all classes ---"
run f_prep_all \
  env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
    --readers BEN13R --all-classes --split random --tag ben13r_all
D=$(ls -d prepared_data/random_top*_ben13r_all 2>/dev/null | head -1)
if [ -z "$D" ]; then say "SKIP f_train_all — no prepared dir"; else
  run f_train_all \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir "$D" --epochs 50 --patience 10
fi

for f in R1 R2 R8; do
  run f_prep_all_loro_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers BEN13R R1 R2 R8 --all-classes --split loro --holdout $f \
      --max-train 40000 --tag allwithben13
  DL=$(ls -d prepared_data/loro_${f}_top*_cap40000_allwithben13 2>/dev/null | head -1)
  if [ -z "$DL" ]; then say "SKIP f_train_all_loro_$f — no prepared dir"; else
    run f_train_all_loro_$f \
      env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
        --data-dir "$DL" --epochs 50 --patience 10
  fi
done

# ---------------------------------------------------------------------------
# PART B — the decisive test. Same LORO folds, same 11 classes, but every
# reader capped to the smallest so Ben13 contributes an equal share instead
# of half. If the loss disappears, the corpus was never the problem.
# ---------------------------------------------------------------------------
say "--- PART B: reader-balanced LORO (the confound test) ---"
for f in R1 R2 R8; do
  run f_prep_bal_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers BEN13R R1 R2 R8 --num-classes 11 --split loro --holdout $f \
      --balance-readers --tag balben13
  run f_train_bal_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top11_balben13 --epochs 50 --patience 10
done

# control arm: the same balanced setup WITHOUT Ben13, so the comparison is
# like-for-like rather than against the old unbalanced baseline
for f in R1 R2 R8; do
  run f_prep_bal_noben_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers R1 R2 R8 --num-classes 11 --split loro --holdout $f \
      --balance-readers --tag balnoben
  run f_train_bal_noben_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top11_balnoben --epochs 50 --patience 10
done

say "===== FOLLOW-UP complete ====="
$PY collect_ben13_results.py --json $OUT/summary.json >> $LOG 2>&1
say "SUMMARY updated"
