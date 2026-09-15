#!/bin/bash
# ============================================================================
# Ben13 reading corpus — extraction + full training programme (~30h window)
# ============================================================================
# Waits for the Torah portion of the download to finish, extracts word-level
# features, then trains in the order requested: 11 classes first (comparable
# to every historical number in this project), then all classes. Nevi'im keep
# downloading throughout and are trained last, as extra data from a related
# but not identical reading style.
#
# Conventions match run_48h_sweep.sh: done/ markers, per-step logs, a failure
# never aborts the queue, `touch results/ben13_train/STOP` exits cleanly.
# ============================================================================
set -u
cd "$(dirname "$0")"

PY=venv/bin/python
FEAT=$PWD/features_v3_word
OUT=results/ben13_train
mkdir -p $OUT/done $OUT/logs
LOG=$OUT/PROGRESS.log
started=$(date +%s)

say(){ echo "[$(date '+%m-%d %H:%M')] [+$(( ($(date +%s)-started)/60 ))m] $*" | tee -a $LOG; }
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

say "===== Ben13 reading: extraction + training ====="

# --- wait for the Torah books to finish downloading -------------------------
# The download walks the table in order, so books 1-5 (27,788 segments) all
# land before Nevi'im starts. Poll rather than guess at a duration.
# Gate on the MANIFEST, not on the file count. The downloader writes the
# manifest only when it exits, so counting mp3s on disk said "ready" while
# the manifest still held 150 rows from an earlier trial -- and the whole
# training programme then ran on 361 words instead of 68,609.
say "waiting for the manifest to list the Torah segments..."
while true; do
  stop
  n=$($PY -c "
import json,sys
try:
    r=json.load(open('data/ben13_reading/manifest_yerushalmi.json',encoding='utf-8'))
    print(sum(1 for x in r if x['book'] in '12345'))
except Exception: print(0)
")
  [ "$n" -ge 27000 ] && { say "manifest ready ($n Torah segments)"; break; }
  say "  manifest has $n Torah segments, waiting..."
  sleep 300
done

# --- extract word-level features from the Torah -----------------------------
run x1_extract_torah \
  $PY Data_Prep/extract_ben13_reading_words.py --books 1 2 3 4 5 --out-reader BEN13R

# HARD GATE. Everything below trains ON these features, and
# 00_validate_and_prepare_data.py only WARNS about a missing reader rather
# than failing -- so without this check a failed extraction silently produces
# a full set of training runs on the old readers alone, labelled as if Ben13
# were in them. That is exactly what happened on the previous attempt: four
# hours of GPU measuring nothing. "A failure never aborts the queue" is right
# for independent experiments and wrong when one step produces the data for
# all the others.
if [ ! -f features_v3_word/BEN13R_X.npy ]; then
  say "ABORT: BEN13R features missing — refusing to run training that would"
  say "       silently fall back to the old readers. Fix the extraction first."
  exit 1
fi
say "gate passed: BEN13R features present ($(du -h features_v3_word/BEN13R_X.npy | cut -f1))"

# ---------------------------------------------------------------------------
# TRAINING BLOCK 1 — 11 classes.
# Requested explicitly, and the reason is comparability: every number in this
# project's history -- the 51.2% baseline, the 88.4% Stage A result, the
# BiLSTM's +12.8 -- was measured on the same top-11 vocabulary. Training the
# new corpus on 11 first is the only way to say whether it is better DATA,
# rather than a different question. 21/25 classes answer "how far can we go";
# 11 answers "is this corpus an improvement", which is the actual question.
# ---------------------------------------------------------------------------
say "--- BLOCK 1: 11 classes (comparable to every historical number) ---"
run t1_prep_11 \
  env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
    --readers BEN13R --num-classes 11 --split random --tag ben13r
run t1_train_11 \
  env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
    --data-dir prepared_data/random_top11_ben13r --epochs 50 --patience 10

# cross-reader: train on Ben13's reader, test on the human readers. This is
# the real generalisation question -- a corpus is only better if it transfers.
for f in R1 R2 R8; do
  run t1_prep_11_loro_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers BEN13R R1 R2 R8 --num-classes 11 --split loro --holdout $f \
      --max-train 40000 --tag withben13
  run t1_train_11_loro_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top11_cap40000_withben13 \
      --epochs 50 --patience 10
done

# ---------------------------------------------------------------------------
# TRAINING BLOCK 2 — all classes.
# ---------------------------------------------------------------------------
say "--- BLOCK 2: all classes ---"
run t2_prep_all \
  env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
    --readers BEN13R --all-classes --split random --tag ben13r_all
# Resolve the glob into a variable and CHECK it. Inlining $(...) here was a
# trap: an empty result does not error, it disappears from the argument list,
# so --data-dir swallows the next flag (--epochs) and training runs with
# silently wrong parameters instead of failing.
D_ALL=$(ls -d prepared_data/random_top*_ben13r_all 2>/dev/null | head -1)
if [ -z "$D_ALL" ]; then
  say "SKIP t2_train_all — prep produced no directory"
else
  run t2_train_all \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir "$D_ALL" --epochs 50 --patience 10
fi

for f in R1 R2 R8; do
  run t2_prep_all_loro_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers BEN13R R1 R2 R8 --all-classes --split loro --holdout $f \
      --max-train 40000 --tag allwithben13
  D_L=$(ls -d prepared_data/loro_${f}_top*_cap40000_allwithben13 2>/dev/null | head -1)
  if [ -z "$D_L" ]; then
    say "SKIP t2_train_all_loro_$f — prep produced no directory"
  else
    run t2_train_all_loro_$f \
      env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
        --data-dir "$D_L" --epochs 50 --patience 10
  fi
done

# ---------------------------------------------------------------------------
# TRAINING BLOCK 3 — sequence models on the best checkpoints.
# The BiLSTM was worth +18.2 points on the old corpus. It needs verse
# structure, which this corpus has: segments carry book/perek/pasuk.
# ---------------------------------------------------------------------------
say "--- BLOCK 3: sequence models ---"
for f in R1 R2 R8; do
  CK=results/loro_${f}_top11_cap40000_withben13/deep_cnn/deep_cnn_best.pth
  [ -f "$CK" ] && run t3_seq_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 14_sequence_model.py --checkpoint $CK \
      --readers BEN13R R1 R2 R8 --holdout $f --crf --tag ben13r_crf
done

# ---------------------------------------------------------------------------
# BLOCK 4 — Nevi'im, once they have finished downloading.
# Haftara chanting is the same accent system but a recognisably different
# melodic style. Whether that is useful extra data or harmful domain shift is
# an empirical question, so it is trained separately and compared, never
# silently merged into the Torah numbers.
# ---------------------------------------------------------------------------
say "--- BLOCK 4: waiting for Nevi'im download ---"
while true; do
  stop
  n=$(find data/ben13_reading/yerushalmi -name "*.mp3" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 42000 ] && { say "Nevi'im download complete ($n files)"; break; }
  # don't wait forever if the download died
  pgrep -f download_ben13_reading > /dev/null || { say "downloader gone, proceeding with $n"; break; }
  sleep 300
done

run x2_extract_neviim \
  $PY Data_Prep/extract_ben13_reading_words.py --books 8 9 --out-reader BEN13N

run t4_prep_neviim \
  env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
    --readers BEN13N --num-classes 11 --split random --tag neviim
run t4_train_neviim \
  env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
    --data-dir prepared_data/random_top11_neviim --epochs 50 --patience 10

# BEN13N was already extracted before this run, so block 4 does not depend on
# the downloader still being alive -- but assert it rather than assume.
[ -f features_v3_word/BEN13N_X.npy ] || say "WARN: BEN13N missing, block 4 will train without it"

# Torah + Nevi'im together, tested on the human readers: does the extra,
# stylistically different data help or hurt?
for f in R1 R2 R8; do
  run t4_prep_combined_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers BEN13R BEN13N R1 R2 R8 --num-classes 11 --split loro --holdout $f \
      --max-train 40000 --tag torahneviim
  run t4_train_combined_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top11_cap40000_torahneviim \
      --epochs 50 --patience 10
done

say "===== complete: $(ls $OUT/done | wc -l | tr -d ' ') steps ====="
