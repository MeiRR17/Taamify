#!/bin/bash
# ============================================================================
# Taamify — sweep part 2 (phases 6-11), ~37 hours
# ============================================================================
# Part 1 (run_48h_sweep.sh) measures out at roughly 21 hours against real
# timings: 42 min per deep_cnn run, ~13 min per sequence model, ~45 min per
# resnet. That leaves most of the window unused, so this is the rest — chosen
# for value, not to burn clock.
#
# Chained automatically after part 1 by run_chain.sh. Same conventions:
# done/ markers, per-step logs, failures never abort the queue.
# ============================================================================
set -u
cd "$(dirname "$0")"

PY=venv/bin/python
FEAT_WORD=$PWD/features_v3_word
OUT=results/sweep_48h
mkdir -p $OUT/done $OUT/logs
LOG=$OUT/PROGRESS.log
READERS="R1 R2 R8"
FOLDS="R1 R2 R8"

started=$(date +%s)
say(){ echo "[$(date '+%m-%d %H:%M')] [P2+$(( ($(date +%s)-started)/60 ))m] $*" | tee -a $LOG; }
stop_requested(){ [ -f $OUT/STOP ] && { say "STOP file found — exiting cleanly"; exit 0; }; }
run(){
  local name=$1; shift
  stop_requested
  if [ -f "$OUT/done/$name" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  local t0=$(date +%s)
  if "$@" > "$OUT/logs/$name.log" 2>&1; then
    say "OK    $name  ($(( ($(date +%s)-t0)/60 ))m)"; touch "$OUT/done/$name"
  else
    say "FAIL  $name  ($(( ($(date +%s)-t0)/60 ))m) — see logs/$name.log; continuing"
  fi
}

say "================= PART 2 begins (phases 6-11) ================="

# ---------------------------------------------------------------------------
# PHASE 6 — wav2vec2 frozen embeddings.
# The highest-upside experiment available, and the only one that attacks this
# project's central problem head-on. Every model here reads mel-spectrograms,
# which encode the speaker as much as the melody -- which is exactly why
# unseen-reader accuracy collapses while same-reader accuracy is fine.
# Self-supervised speech encoders are trained to be far more speaker-robust.
# The Hebrew wav2vec2 is already on disk (models/wav2vec2-he, 1.2GB), used
# today only for CTC alignment, never as a feature extractor.
# ---------------------------------------------------------------------------
say "--- PHASE 6: wav2vec2 frozen embeddings as an alternative representation ---"
run p6_extract_w2v \
  $PY Data_Prep/extract_wav2vec_features.py --readers R1 R2 R8
for f in $FOLDS; do
  run p6_prep_w2v_$f \
    env TAAMIFY_FEATURES=$PWD/features_w2v $PY 00_validate_and_prepare_data.py \
      --readers $READERS --num-classes 21 --split loro --holdout $f \
      --max-train 40000 --tag w2v
  run p6_cnn_w2v_$f \
    env TAAMIFY_FEATURES=$PWD/features_w2v $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top21_cap40000_w2v --epochs 50 --patience 10
done

# ---------------------------------------------------------------------------
# PHASE 7 — the R1 wall, tested rather than assumed.
# R1 held out sits at ~42% in every configuration ever run, while R2 and R8
# reach 70-83%. Milestone 1 proposed an explanation -- R1 covers 91% of the
# Torah, so a pool without it is narrow -- but that was inferred from
# coverage numbers, never tested by varying coverage directly. This does:
# train R1-heldout with progressively fewer training words and see whether
# accuracy tracks the amount of data, or is flat (which would mean the wall
# is about R1's voice, not the corpus size).
# ---------------------------------------------------------------------------
say "--- PHASE 7: is the R1 wall about coverage, or about R1's voice? ---"
for cap in 10000 20000 30000 40000; do
  run p7_prep_cap$cap \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 00_validate_and_prepare_data.py \
      --readers $READERS --num-classes 21 --split loro --holdout R1 \
      --max-train $cap --tag capsweep
  run p7_cnn_cap$cap \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_R1_top21_cap${cap}_capsweep --epochs 50 --patience 10
done

# ---------------------------------------------------------------------------
# PHASE 8 — 25 classes, so the app can cover every ta'am that exists.
# 21 classes leaves 11 Torah words unverifiable (Mercha_Kefula, Shalshelet,
# Yerach_Ben_Yomo, Qarney_Para). Trivial in count, but "the app handles every
# ta'am" is a different claim from "the app handles almost every ta'am", and
# the only way to know the cost is to measure it.
# ---------------------------------------------------------------------------
say "--- PHASE 8: all 25 classes ---"
for f in $FOLDS; do
  run p8_prep_25_$f \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 00_validate_and_prepare_data.py \
      --readers $READERS --all-classes --split loro --holdout $f \
      --max-train 40000 --tag all25
  run p8_cnn_25_$f \
    env TAAMIFY_FEATURES=$FEAT_WORD $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir $(ls -d prepared_data/loro_${f}_top*_cap40000_all25 2>/dev/null | head -1) \
      --epochs 50 --patience 10
done

# ---------------------------------------------------------------------------
# PHASE 9 — sequence models with the grammar prior, at 21 classes.
# 14_sequence_model.py has a --grammar-prior flag that biases the CRF's
# transition matrix toward the near-deterministic ta'am pairs. It returned a
# null result (+0.1) at 11 classes, single seed. At 21 classes there are more
# constrained transitions for it to exploit, and now there are seeds.
# ---------------------------------------------------------------------------
say "--- PHASE 9: CRF grammar prior at 21 classes, multi-seed ---"
for f in $FOLDS; do
  CK=results/loro_${f}_top21_cap40000/deep_cnn/deep_cnn_best.pth
  for seed in 0 1 2; do
    run p9_gprior_${f}_s$seed \
      env TAAMIFY_FEATURES=$FEAT_WORD TAAMIFY_SEED=$seed $PY 14_sequence_model.py \
        --checkpoint $CK --readers $READERS --holdout $f --crf --grammar-prior \
        --tag s${seed}_gprior
  done
done

# ---------------------------------------------------------------------------
# PHASE 10 — more seeds on the headline result.
# Part 1 runs seeds 1-2. Three draws is enough to see a spread but not to
# trust one; five gives a usable standard deviation, which is what every
# number in results/v3_comparison/ has been missing.
# ---------------------------------------------------------------------------
say "--- PHASE 10: seeds 3-5 on the headline LORO result ---"
for f in $FOLDS; do
  for seed in 3 4 5; do
    run p10_seed${seed}_$f \
      env TAAMIFY_FEATURES=$FEAT_WORD TAAMIFY_SEED=$seed $PY 10_train_pipeline.py \
        --model deep_cnn --data-dir prepared_data/loro_${f}_top21_cap40000 \
        --epochs 50 --patience 10 --tag seed$seed
  done
done

# ---------------------------------------------------------------------------
# PHASE 11 — learning rate / weight decay on the best-known config.
# Lowest expected value of anything here (the architecture was never the
# bottleneck; the reader gap is), which is exactly why it goes last: it fills
# remaining time without displacing an experiment that answers a real
# question.
# ---------------------------------------------------------------------------
say "--- PHASE 11: hyperparameters (lowest priority, fills remaining time) ---"
for lr in 0.0002 0.001; do
  for wd in 1e-4 1e-2; do
    for f in R1 R2; do
      run p11_lr${lr}_wd${wd}_$f \
        env TAAMIFY_FEATURES=$FEAT_WORD $PY 10_train_pipeline.py --model deep_cnn \
          --data-dir prepared_data/loro_${f}_top21_cap40000 --epochs 50 --patience 10 \
          --lr $lr --weight-decay $wd --tag lr${lr}_wd${wd}
    done
  done
done

say "================= PART 2 complete ================="
say "total steps done: $(ls $OUT/done 2>/dev/null | wc -l | tr -d ' ')"
