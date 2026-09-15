#!/bin/bash
# Boundary-quality ablation (FINDINGS.md §8's pending item): does dropping
# "stored" rows (Whisper-fallback boundaries verse-level CTC never replaced)
# improve LORO accuracy? R1 has the highest stored-share of the three main
# readers (36% vs R2's 23%/R8's 14%) -- a direct test of whether R1's wall
# is partly boundary noise, not purely voice/generalization.
#
# Data prep (00_validate_and_prepare_data.py --exclude-align stored) is
# already done for all 3 LORO folds -- CPU work, ran while the wav2vec2
# sequence probe's extraction was still going. Only the GPU training step
# waits here, and it waits for the sequence probe's *training* to finish
# too (not just its extraction), so GPU stays strictly serial project-wide.
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
FEAT=$PWD/features_v3_word
OUT=results/boundary_ablation
mkdir -p $OUT/done $OUT/logs
LOG=$OUT/PROGRESS.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
run(){
  local name=$1; shift
  [ -f "$OUT/done/$name" ] && { say "SKIP $name"; return 0; }
  say "START $name"; local t0=$(date +%s)
  if "$@" > "$OUT/logs/$name.log" 2>&1; then
    say "OK    $name ($(( ($(date +%s)-t0)/60 ))m)"; touch "$OUT/done/$name"
  else
    say "FAIL  $name ($(( ($(date +%s)-t0)/60 ))m) — see logs/$name.log"
  fi
}

say "===== boundary-quality ablation ====="
say "waiting for prepared_data/loro_R8_top11_noalign (background prep) ..."
while [ ! -f prepared_data/loro_R8_top11_noalign/metadata.json ]; do sleep 15; done
say "R8 prep done; all 3 folds' data ready"

say "waiting for wav2vec2 sequence probe to finish (GPU serial) ..."
while [ ! -f results/wav2vec_seq_probe/summary.json ]; do sleep 60; done
say "GPU free"

for f in R1 R2 R8; do
  run train_noalign_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top11_noalign --epochs 50 --patience 10
done

say "===== ablation training complete — see results/loro_*_top11_noalign/deep_cnn/deep_cnn_metrics.json ====="
