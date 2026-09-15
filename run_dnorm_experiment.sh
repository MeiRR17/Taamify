#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
OUT=results/dnorm
mkdir -p $OUT/done $OUT/logs
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $OUT/PROGRESS.log; }
say "===== duration-normalised features experiment ====="
[ -f $OUT/done/build ] || {
  say "building duration-normalised features"
  $PY Data_Prep/make_duration_normalized_features.py --readers R1 R2 R8 \
    > $OUT/logs/build.log 2>&1 && touch $OUT/done/build
}
[ -f features_v3_dnorm/R8_X.npy ] || { say "ABORT: build failed"; exit 1; }
export TAAMIFY_FEATURES=$PWD/features_v3_dnorm
for f in R1 R2 R8; do
  [ -f $OUT/done/prep_$f ] || {
    say "prep $f"
    $PY 00_validate_and_prepare_data.py --readers R1 R2 R8 --num-classes 11 \
      --split loro --holdout $f --tag dnorm > $OUT/logs/prep_$f.log 2>&1 && touch $OUT/done/prep_$f
  }
  [ -f $OUT/done/train_$f ] && { say "SKIP train_$f"; continue; }
  [ -d prepared_data/loro_${f}_top11_dnorm ] || { say "no data for $f"; continue; }
  say "START train_$f"; t0=$(date +%s)
  if $PY 10_train_pipeline.py --model deep_cnn \
       --data-dir prepared_data/loro_${f}_top11_dnorm --epochs 50 --patience 10 \
       > $OUT/logs/train_$f.log 2>&1; then
    say "OK train_$f ($(( ($(date +%s)-t0)/60 ))m)"; touch $OUT/done/train_$f
  else say "FAIL train_$f"; fi
done
say "===== dnorm experiment done ====="
