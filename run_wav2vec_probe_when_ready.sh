#!/bin/bash
# Waits for (1) the wav2vec2 mean-pool extraction and (2) the phrase-realign
# sweep's GPU work to both finish, then trains the probe. GPU/MPS training
# stays serial project-wide -- this just enforces that ordering automatically
# instead of needing a manual check-back.
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
LOG=results/wav2vec_probe/logs/wait.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

say "waiting for wav2vec2 extraction (features_w2v_pool/R8_meta.json) ..."
while [ ! -f features_w2v_pool/R8_meta.json ]; do sleep 30; done
say "extraction done"

say "waiting for phrase-realign sweep to finish (results/phrase_realign/done/p_train_loro_R8) ..."
while [ ! -f results/phrase_realign/done/p_train_loro_R8 ] && [ ! -f results/phrase_realign/STOP ]; do
  sleep 60
done
say "phrase-realign sweep done (or stopped) -- GPU free"

say "=== training wav2vec2 probe ==="
$PY Data_Prep/train_wav2vec_probe.py > results/wav2vec_probe/logs/train.log 2>&1
say "=== probe training complete, see results/wav2vec_probe/summary.json ==="
