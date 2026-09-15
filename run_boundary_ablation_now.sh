#!/bin/bash
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
FEAT=$PWD/features_v3_word
OUT=results/boundary_ablation
mkdir -p $OUT/done $OUT/logs
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $OUT/PROGRESS.log; }
say "===== boundary ablation: running NOW, no GPU gate ====="
for f in R1 R2 R8; do
  [ -f "$OUT/done/train_noalign_$f" ] && { say "SKIP $f"; continue; }
  # R8's prep may still be finishing in the background
  while [ ! -f prepared_data/loro_${f}_top11_noalign/metadata.json ]; do sleep 10; done
  say "START train_noalign_$f"; t0=$(date +%s)
  if env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
       --data-dir prepared_data/loro_${f}_top11_noalign --epochs 50 --patience 10 \
       > "$OUT/logs/train_noalign_$f.log" 2>&1; then
    say "OK    train_noalign_$f ($(( ($(date +%s)-t0)/60 ))m)"; touch "$OUT/done/train_noalign_$f"
  else
    say "FAIL  train_noalign_$f — see $OUT/logs/train_noalign_$f.log"
  fi
done
say "===== boundary ablation complete ====="
