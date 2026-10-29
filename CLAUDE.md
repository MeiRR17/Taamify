# Taamify — project instructions

## Agent continuation / פקודת "תמשיך" (documentation and safe work only)

When the user asks to continue Taamify, read `docs/research-os/CONTINUE_AGENT_PROMPT_HE.md`, `MASTER_COMPLETION_PLAN_HE.md` and `CONTINUATION_STATE.md`; follow their evidence-first, non-destructive continuation protocol. This supplements the existing product and measurement constraints below. Before resuming, reconcile local unpushed reports with GitHub Issue #4 and PR #5; don't overwrite another agent's work. A bare "תמשיך" does not authorize prod checkpoint or threshold changes, merging, deployment, payment, public audio transfer, or large training sweeps.


Deep learning that identifies **טעמי מקרא** (cantillation marks) from audio,
for the **Sephardic-Jerusalem tradition only**, aimed at a real product for
bar-mitzvah learners. Not a research comparison.

## Scope — non-negotiable

- **One tradition: Sephardic-Jerusalem.** Ashkenazi, Moroccan, Yemenite audio
  is not extra data — it is *mislabeled* data here, because the melodies
  genuinely differ. ben13.co.il also serves `ashkenazi` and `maroko`; both
  were downloaded once and deleted on instruction. Other traditions appear on
  the roadmap as a future market, never as current work.
- **One corpus: Ben13.** Measured head to head, adding the three YouTube
  readers (R1/R2/R8, 80,285 words) *lowered* accuracy by 4.3 points. Noisy
  alignment does not average out; it smears over the clean data. R1/R2/R8
  features still exist on disk — do not train on them.
- All user-facing text is **Hebrew, RTL**. This includes dashboards, the
  recording studio, and the learning app.

## Run it

```bash
venv/bin/python 10_train_pipeline.py --model deep_cnn --data-dir prepared_data/<scenario>
venv/bin/python 00_validate_and_prepare_data.py --readers BEN13R --num-classes 21 --split parasha
venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
GPU_SLOTS=6 CPU_SLOTS=4 MIN_FREE_GB=6 ./run_queue.sh &     # the job queue
venv/bin/python build_ops_dashboard.py                      # live status page
venv/bin/python export_knowledge_base.py                    # Obsidian vault
```

`TAAMIFY_FEATURES` must point at `features_v3_word` — the default is the old
`features_v2` directory and a job that forgets it fails with a confusing
`FileNotFoundError` on `BEN13R_X.npy`.

## Measurement rules — these have already caught wrong conclusions

- **The seed standard deviation is 1.17 points** (measured, 5 seeds, same
  config). **Any difference smaller than ~2.3 points is noise.** Several early
  "improvements" in this project do not survive this bar. Never report a
  single-seed delta as a finding.
- **A random split leaks.** Ben13 is one voice: neighbouring words of the same
  verse share a recording, a microphone and often a breath. The honest number
  comes from `--split parasha`, holding out whole parashot. Current gap
  between the two: ~1 point.
- **For imbalanced tasks, plain accuracy is meaningless.** 79.5% of words are
  milra, so a model that always answers "milra" scores 79.5%. Report
  **balanced accuracy**; the trivial baseline there is 50%.
- **Always run the control.** A metric that improves can improve for the wrong
  reason. Shorter windows score better *because* they are shorter; a model can
  separate ayin-words from alef-words by vocabulary rather than sound. Shuffled
  labels and same-width random windows are the controls that caught these.
- **Measure cheap before building expensive.** Station 4 (stress) was budgeted
  at a week plus two weeks for a prerequisite. Relabelling the *existing*
  spectrograms answered it in a day, and the prerequisite turned out to be
  unnecessary. Ask what the cheapest measurement is that could kill the idea.
- **A result that looks too good is almost always a bug.** Three "independent"
  parasha draws returning the identical number to four decimals was a fixed
  RNG seed, not a stable model.

## Product wiring — the failure that already happened

234 GPU-hours of model work sat on disk while the live app served an older
checkpoint, and then broke outright (`500` on `/api/verify_reading`). **A model
that is not wired into `webapp/server.py` does not exist as far as the product
is concerned.**

- The verify head is trained **against a specific CNN's embedding space**.
  Retraining the CNN invalidates it. `get_verify_head()` asserts the class
  names match — that assert is the last line of defence, and it fires as a
  500, so check after any checkpoint swap.
- `build_ops_dashboard.py` parses `webapp/server.py` to report what is
  *actually* served. Use it rather than assuming.

## Job queue conventions

`run_queue.sh` keeps the machine full: `jobs/queue/NNN_name[.gpu].job` →
`running/` → `done/` | `failed/`, with logs in `jobs/logs/`.

- `.gpu.job` goes to the GPU pool, anything else to the CPU pool. The GPU is
  **one device** — a dozen concurrent MPS trainings just timeshare it and sit
  at ~4% CPU each while half the machine idles. Six GPU + four CPU is the
  measured sweet spot.
- Declare inputs with `# NEEDS: <path>` instead of waiting inside the job.
  In-job wait loops hold a slot doing nothing and deadlock when the producing
  job is queued behind the waiter.
- **macOS ships bash 3.2.** `BASHPID` does not exist there, and under `set -u`
  reading it kills the subshell instantly — this logged START for six jobs and
  ran none of them for 13 hours. Check bash-3.2 compatibility in queue code.
- After killing a pool parent, check for orphans:
  `ps -eo pid,ppid,command | awk '$2==1'`. `ProcessPoolExecutor` workers
  routinely survive and hold several GB each. This has recurred three times.
- Cap threads per worker (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`) —
  otherwise N workers each spawn ~10 threads on a 10-core machine and more
  workers run *slower*.

## Never commit

- `app_assets/collected_attempts/` — other people's voice recordings.
- `incoming/` — exported learner recordings.
- Audio corpora (`data/ben13_reading/`, ~1.9 GB) and `features_*/`.

## Current state

Torah model 86.9% (parashot held out) · sequence model 94.6% · verify head
97.7% · stress detection 92.5% balanced. **Every one of those numbers comes
from a single voice.** Live check: 15/16 correct on Ben13's own clips, 0/2 on
another reader, at high confidence. Until a second Sephardic-Jerusalem voice
is recorded, it is unknown whether the model learned the ta'amim or learned
one reader — treat that as the project's open risk, not a footnote.
