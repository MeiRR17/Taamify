#!/bin/bash
# Waits for extract_wav2vec_sequence.py to finish, then trains the
# sequence-aware (BiLSTM + attention) probe. GPU is already free (the
# phrase-realign sweep and the mean-pool probe both finished earlier), so
# unlike run_wav2vec_probe_when_ready.sh this only needs to gate on extraction.
set -u
cd "$(dirname "$0")"
PY=venv/bin/python
LOG=results/wav2vec_seq_probe/logs/wait.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

say "waiting for wav2vec2 sequence extraction (features_w2v_seq/R8_meta.json) ..."
while [ ! -f features_w2v_seq/R8_meta.json ]; do sleep 30; done
say "extraction done"

say "=== training wav2vec2 sequence probe (BiLSTM + attention) ==="
$PY Data_Prep/train_wav2vec_seq_probe.py > results/wav2vec_seq_probe/logs/train.log 2>&1
say "=== probe training complete, see results/wav2vec_seq_probe/summary.json ==="
