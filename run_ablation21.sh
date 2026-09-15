#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
export TAAMIFY_FEATURES=$PWD/features_v3_word
OUT=results/boundary_ablation
mkdir -p $OUT/done $OUT/logs
say(){ echo "[$(date '+%m-%d %H:%M')] [21cls] $*" | tee -a $OUT/PROGRESS.log; }
say "===== 21-class boundary ablation ====="
for f in R1 R2 R8; do
  [ -f "$OUT/done/prep21_$f" ] || {
    say "prep $f"
    $PY 00_validate_and_prepare_data.py --readers R1 R2 R8 --num-classes 21 \
      --split loro --holdout $f --exclude-align stored --tag noalign21 \
      > "$OUT/logs/prep21_$f.log" 2>&1 && touch "$OUT/done/prep21_$f"
  }
done
for f in R1 R2 R8; do
  [ -f "$OUT/done/train21_$f" ] && { say "SKIP train21_$f"; continue; }
  [ -d prepared_data/loro_${f}_top21_noalign21 ] || { say "no data for $f, skip"; continue; }
  say "START train21_$f"; t0=$(date +%s)
  if $PY 10_train_pipeline.py --model deep_cnn \
       --data-dir prepared_data/loro_${f}_top21_noalign21 --epochs 50 --patience 10 \
       > "$OUT/logs/train21_$f.log" 2>&1; then
    say "OK train21_$f ($(( ($(date +%s)-t0)/60 ))m)"; touch "$OUT/done/train21_$f"
  else say "FAIL train21_$f"; fi
done
say "===== 21-class ablation done ====="
