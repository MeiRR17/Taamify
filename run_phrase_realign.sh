#!/bin/bash
# ============================================================================
# Phrase-realignment experiment: does short-span CTC (Ben13's structural
# advantage) help R1/R2/R8 too, without recording anyone new?
# ============================================================================
set -u
cd "$(dirname "$0")"

PY=venv/bin/python
FEAT=$PWD/features_v3_word
OUT=results/phrase_realign

# 5 workers hit 63GB with 69MB free (one CTC model per worker, ~4.3GB each)
# and killing this script mid-extraction left 5 of them orphaned (parent=1)
# still holding that memory -- same failure class as the Ben13 run. 3 leaves
# real headroom.
export TAAMIFY_WORKERS=3
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

say "===== phrase-realignment experiment ====="

# --- extraction, one reader at a time (each is CPU-bound + I/O, no point
# running them concurrently on 5-worker pools each) ---
for r in R1 R2 R8; do
  run x_realign_$r $PY Data_Prep/realign_by_phrase.py --readers $r
done

# HARD GATE, same lesson as the Ben13 run: don't let training silently fall
# back to originals if extraction failed.
for r in R1 R2 R8; do
  if [ ! -f features_v3_word/${r}P_X.npy ]; then
    say "ABORT: ${r}P features missing"; exit 1
  fi
done
say "gate passed: R1P/R2P/R8P all present"

# ---------------------------------------------------------------------------
# Direct comparison 1: each reader alone, phrase-realigned vs original,
# random split, 11 classes -- isolates alignment quality from everything else.
# ---------------------------------------------------------------------------
say "--- per-reader: original vs phrase-realigned ---"
for r in R1 R2 R8; do
  run p_prep_orig_$r \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers $r --num-classes 11 --split random --tag orig_$r
  run p_train_orig_$r \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/random_top11_orig_$r --epochs 50 --patience 10

  run p_prep_phrase_$r \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers ${r}P --num-classes 11 --split random --tag phrase_$r
  run p_train_phrase_$r \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/random_top11_phrase_$r --epochs 50 --patience 10
done

# ---------------------------------------------------------------------------
# Direct comparison 2: the real question. LORO with phrase-realigned readers
# instead of the originals -- does R1's wall move?
# ---------------------------------------------------------------------------
say "--- LORO: phrase-realigned R1P/R2P/R8P, holding out the ORIGINAL of each ---"
for f in R1 R2 R8; do
  # hold out the ORIGINAL alignment of the target reader (so the test set is
  # exactly the same words used everywhere else in the project), train on
  # the PHRASE-realigned version of the other two
  others=$(for x in R1P R2P R8P; do [ "$x" != "${f}P" ] && echo -n "$x "; done)
  run p_prep_loro_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 00_validate_and_prepare_data.py \
      --readers $others $f --num-classes 11 --split loro --holdout $f \
      --max-train 40000 --tag phraseloro
  run p_train_loro_$f \
    env TAAMIFY_FEATURES=$FEAT $PY 10_train_pipeline.py --model deep_cnn \
      --data-dir prepared_data/loro_${f}_top11_cap40000_phraseloro \
      --epochs 50 --patience 10
done

say "===== phrase-realignment experiment complete ====="
