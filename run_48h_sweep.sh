#!/bin/bash
# ============================================================================
# Taamify — 48-hour unattended training sweep
# ============================================================================
# Runs the whole queue back to back on MPS, logging every step. Designed to
# survive without supervision: each phase is independent, a failure in one
# does not stop the next, and every result lands on disk as it completes.
#
# HARD REQUIREMENT: the machine must not sleep. Start this under caffeinate:
#   caffeinate -dims ./run_48h_sweep.sh
#
# Watch progress:  tail -f results/sweep_48h/PROGRESS.log
# Kill cleanly:    touch results/sweep_48h/STOP   (finishes current step, exits)
# ============================================================================
set -u
cd "$(dirname "$0")"

PY=venv/bin/python
FEAT_WORD=$PWD/features_v3_word
FEAT_STRESS=$PWD/features_v3_stressed
FEAT_SPKNORM=$PWD/features_v3_word_spknorm
OUT=results/sweep_48h
mkdir -p $OUT
LOG=$OUT/PROGRESS.log
READERS="R1 R2 R8"
FOLDS="R1 R2 R8"

started=$(date +%s)
say(){ echo "[$(date '+%m-%d %H:%M')] [+$(( ($(date +%s)-started)/60 ))m] $*" | tee -a $LOG; }
stop_requested(){ [ -f $OUT/STOP ] && { say "STOP file found — exiting cleanly"; exit 0; }; }

# run <name> <command...>  — never aborts the sweep on failure
run(){
  local name=$1; shift
  stop_requested
  if [ -f "$OUT/done/$name" ]; then say "SKIP $name (already done)"; return 0; fi
  mkdir -p $OUT/done $OUT/logs
  say "START $name"
  local t0=$(date +%s)
  if "$@" > "$OUT/logs/$name.log" 2>&1; then
    say "OK    $name  ($(( ($(date +%s)-t0)/60 ))m)"
    touch "$OUT/done/$name"
  else
    say "FAIL  $name  ($(( ($(date +%s)-t0)/60 ))m) — see logs/$name.log; continuing"
  fi
}

say "================= 48h sweep begins ================="
say "M1 Max · 64GB unified · MPS. One job at a time: the GPU is the bottleneck,"
say "so parallelism would only cause memory contention, not speed."

# ---------------------------------------------------------------------------
# PHASE 0 — prep: R8 is missing from two feature variants, so the
# representation comparison in phase 2 cannot run without extracting it first.
# ---------------------------------------------------------------------------
say "--- PHASE 0: extract the missing R8 feature variants ---"
run p0_extract_stressed_R8 \
  $PY Data_Prep/extract_features_v3.py --window stressed --readers R8
run p0_extract_spknorm_R8 \
  $PY Data_Prep/extract_features_v3.py --window word --speaker-norm --readers R8

# ---------------------------------------------------------------------------
# PHASE 1 — the main event.
# The BiLSTM was the single biggest lever ever measured here (+12.8 points at
# 11 classes, Milestone 2) but was never re-measured at 21. If the lift holds,
# LORO goes from ~59.5% to roughly 70%. Three folds, three configurations,
# three seeds — because the CRF's +0.4 has always been single-seed and nobody
# knows yet whether it is real.
# ---------------------------------------------------------------------------
say "--- PHASE 1: LORO CNNs at 21 classes, then sequence models ---"
for f in $FOLDS; do
  run p1_prep_loro_$f \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 00_validate_and_prepare_data.py \
      --readers $READERS --num-classes 21 --split loro --holdout $f --max-train 40000
  run p1_cnn_loro_$f \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top21_cap40000 --epochs 50 --patience 10
done

for f in $FOLDS; do
  CK=results/loro_${f}_top21_cap40000/deep_cnn/deep_cnn_best.pth
  for seed in 0 1 2; do
    run p1_seq_${f}_plain_s$seed \
      env TAAMIFY_FEATURES=$FEAT_WORD TAAMIFY_SEED=$seed $PY 14_sequence_model.py \
        --checkpoint $CK --readers $READERS --holdout $f --tag s${seed}_plain
    run p1_seq_${f}_crf_s$seed \
      env TAAMIFY_FEATURES=$FEAT_WORD TAAMIFY_SEED=$seed $PY 14_sequence_model.py \
        --checkpoint $CK --readers $READERS --holdout $f --crf --tag s${seed}_crf
  done
done

# ---------------------------------------------------------------------------
# PHASE 2 — representation comparison, all at 21 classes on identical folds.
# FINDINGS.md rejected the stressed-syllable window on melodic-activity
# grounds, but that was measured at 11 classes and on a different metric than
# downstream accuracy. Milestone 9 then found two classes where a stressed
# window genuinely survives a null control. This settles it on accuracy.
# ---------------------------------------------------------------------------
say "--- PHASE 2: word vs stressed vs speaker-normalised features ---"
for f in $FOLDS; do
  # --tag keeps each feature source in its OWN prepared_data dir. Without it
  # all three variants collide on one name and later phases silently train on
  # whichever was written last.
  run p2_prep_stress_$f \
    env TAAMIFY_FEATURES=$FEAT_STRESS $PY 00_validate_and_prepare_data.py \
      --readers $READERS --num-classes 21 --split loro --holdout $f \
      --max-train 40000 --tag stressed
  run p2_cnn_stress_$f \
    env TAAMIFY_FEATURES=$FEAT_STRESS $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top21_cap40000_stressed \
      --epochs 50 --patience 10
  run p2_prep_spknorm_$f \
    env TAAMIFY_FEATURES=$FEAT_SPKNORM $PY 00_validate_and_prepare_data.py \
      --readers $READERS --num-classes 21 --split loro --holdout $f \
      --max-train 40000 --tag spknorm
  run p2_cnn_spknorm_$f \
    env TAAMIFY_FEATURES=$FEAT_SPKNORM $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top21_cap40000_spknorm \
      --epochs 50 --patience 10
done

# ---------------------------------------------------------------------------
# PHASE 3 — architecture sweep at 21 classes. deep_cnn beat resnet18_tl at 11
# classes (Milestone 0) and became the default on that basis; nobody has
# checked whether that still holds with twice the classes, or how the
# sequence-native architectures do.
# ---------------------------------------------------------------------------
say "--- PHASE 3: architectures at 21 classes ---"
for f in $FOLDS; do
  for arch in resnet18_tl crnn_lstm transformer; do
    run p3_${arch}_$f \
      env TAAMIFY_FEATURES=$FEAT_WORD $PY 10_train_pipeline.py --model $arch \
        --data-dir prepared_data/loro_${f}_top21_cap40000 --epochs 50 --patience 10
  done
done

# ---------------------------------------------------------------------------
# PHASE 4 — multi-seed the headline number itself. Every LORO figure quoted in
# this project is single-seed. Without a spread, "R2 82.8%" is one draw.
# ---------------------------------------------------------------------------
say "--- PHASE 4: seed spread on the headline deep_cnn LORO result ---"
for f in $FOLDS; do
  for seed in 1 2; do
    run p4_seed${seed}_$f \
      env TAAMIFY_FEATURES=$FEAT_WORD TAAMIFY_SEED=$seed $PY 10_train_pipeline.py \
        --model deep_cnn --data-dir prepared_data/loro_${f}_top21_cap40000 \
        --epochs 50 --patience 10 --tag seed$seed
  done
done

# ---------------------------------------------------------------------------
# PHASE 5 — Stage A verify heads for everything trained above. Cheap (a small
# MLP on frozen embeddings) but it is the number the product actually depends
# on, so every checkpoint should have one.
# ---------------------------------------------------------------------------
say "--- PHASE 5: Stage A verify heads on every checkpoint ---"
for f in $FOLDS; do
  CK=results/loro_${f}_top21_cap40000/deep_cnn/deep_cnn_best.pth
  run p5_verify_$f \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 15_binary_verify_model.py \
      --checkpoint $CK --readers $READERS --holdout $f
done

say "================= sweep complete ================="
say "done: $(ls $OUT/done 2>/dev/null | wc -l | tr -d ' ') steps"
say "failures: $(grep -c '^\[.*FAIL' $LOG || true)"
