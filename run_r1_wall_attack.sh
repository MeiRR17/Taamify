#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
export TAAMIFY_FEATURES=$PWD/features_v3_word
OUT=results/r1_wall
mkdir -p $OUT/done $OUT/logs
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $OUT/PROGRESS.log; }
say "===== R1 wall attack ====="

# matched same-day baseline (stored rows KEPT) — so the ablation comparison
# isn't against a months-old constant from possibly different feature code
[ -f $OUT/done/prep_std ] || {
  say "prep loro_R1_top11_std (baseline, stored kept)"
  $PY 00_validate_and_prepare_data.py --readers R1 R2 R8 --num-classes 11 \
    --split loro --holdout R1 --tag std > $OUT/logs/prep_std.log 2>&1 && touch $OUT/done/prep_std
}
D=prepared_data/loro_R1_top11_std
[ -d "$D" ] || { say "ABORT: no baseline data"; exit 1; }

# config sweep on the one fold that has never moved
run_cfg(){
  local name=$1; shift
  [ -f "$OUT/done/$name" ] && { say "SKIP $name"; return 0; }
  say "START $name"; local t0=$(date +%s)
  if $PY 10_train_pipeline.py --data-dir "$D" --epochs 50 --patience 10 "$@" \
       > "$OUT/logs/$name.log" 2>&1; then
    say "OK $name ($(( ($(date +%s)-t0)/60 ))m)"; touch "$OUT/done/$name"
  else say "FAIL $name"; fi
}
run_cfg base      --model deep_cnn --tag base
run_cfg lr_high   --model deep_cnn --lr 0.002 --tag lrhigh
run_cfg no_specaug --model deep_cnn --no-specaugment --tag nospec
run_cfg wd_high   --model deep_cnn --weight-decay 0.01 --tag wdhigh
run_cfg crnn      --model crnn_lstm --tag crnn
say "===== R1 wall attack done ====="
