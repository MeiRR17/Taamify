# Taamify — שעות אימון, שעות מחקר ותפוקת החומרה

**Version:** 2026-10-21 · **Status:** capacity plan, NOT measured training time or academic timesheet. The live Dvir OS UI `/research/taamify` calculates accelerator hours when the user enters **measured** minutes per epoch and setup hours per pipeline.

## Devices from the canonical Dvir OS v0.9 specification

| Host | Known specification | Research scheduling role | Must still measure |
|---|---|---|---|
| MacBook Pro main | Apple M1 Max, 10 CPU cores, 32 GPU cores, 64GB unified memory | Main MPS experiments; start with 1 heavy GPU job at a time | per-model RAM peak, epochs/minute, thermals, sleep, disk, available GPU hours |
| MacBook Air portable | Apple M4, 16GB unified memory | Developer smoke/prototyping and smaller CPU/inference tests | feasibility; avoid sustained always-on jobs |
| Windows desktop | Ryzen 5 3600, 32GB RAM; GPU text recorded as '16 60 Super' but **unverified** | No CUDA allocation until device ID, VRAM and drivers checked | `nvidia-smi`, precise model/VRAM, CUDA compatibility |
| Other devices | phones/tablets | Capture/UX validation, not training plan | microphone variance and upload/access consent |

**Never promise 24/7 operation:** main server is a MacBook Pro without documented UPS; checkpoint and restartable queues required. Don't run simultaneous 6 heavy MPS jobs merely because the existing queue has 6 slots.

## Computational hours: method (for every pipeline independently)

For pipeline i:

`accelerator_hours_i = (number_of_fits_i × epochs_per_fit_i × measured_minutes_per_epoch_i / 60) + measured_feature_extraction_hours_i + measured_conversion_hours_i + measured_validation_hours_i`

Recorded operational values: `device_id, pipeline_id, training_start, training_end, epochs_completed, peak_unified_memory_gb, mean_epoch_seconds, startup_seconds, model_checkpoint, energy_kwh_if_metered, failure_reason`. Logs must use wall time and idle resources separately. Cache frozen encoder embeddings by exact `encoder_sha + preprocessing_hash + manifest_sha`.

**Calibration measurement:** run a 2–3 epoch warm-up per pipeline on the actual data/batch/device; exclude first-epoch compilation/cache warm-up from steady-state epoch estimate but retain it as setup time. Repeat ≥2 times if variable. For large SSL models that cannot run within RAM, stop at feasibility stage; do not fill a guessed duration.

## Fit budget by pipeline — planning counts, not completed runs

| ID | Method | Planned fits | Planned epochs per fit | Fit-epochs | Eligibility |
|---|---|---:|---:|---:|---|
| P00 | DeepCNN baseline | 3 | 30 | 90 | audit first |
| P01 | F0 / prosody | 3 | 20 | 60 | current data |
| P02 | multiresolution | 3 | 30 | 90 | current data |
| P03 | WavLM frozen embeddings | 3 | 20 | 60 | encoder extraction |
| P04 | W2v-BERT embeddings | 3 | 20 | 60 | memory validation |
| P05 | MuQ | 3 | 20 | 60 | research-only licence |
| P06 | CNN + F0 fusion | 3 | 30 | 90 | paired baseline |
| P07 | teacher verification | 3 | 20 | 60 | real human errors |
| P08 | VC synthetic augmentation | 3 | 30 | 90 | voice consent + fidelity |
| P09 | adversarial voice invariance | 9 | 30 | 270 | multiple genuine readers |
| P10 | contrastive voice invariance | 9 | 30 | 270 | multiple genuine readers |
| P11 | adapter or LoRA | 3 | 20 | 60 | finalist encoder |
| P12 | distillation | 3 | 30 | 90 | validated winner |
| **Total theoretical ceiling** | | **51** | | **1,350** | these are **not** all runnable today |

**Important:** P00–P12 catalog describes a scenario grid rather than a schedule that can immediately be executed. P09/P10's nine fits anticipate three seeds × three development folds, but final held-out speaker validation is separate. A stage-gated approach saves computational time by declining to run unpromising or infeasible candidates.

Illustration only: if **every** fit-epoch, implausibly, took exactly two minutes, then 1,350 × 2 ÷ 60 = **45 accelerator-hours** for training alone; at five minutes each = **112.5 hours**. These figures exclude encoder embedding extraction, voice conversion, eval, restarts, disk IO and debugging, and do NOT represent a measured estimate or imply different encoders run equally fast.

## Human research hours ≠ training hours

| Phase | Academic person-hours | Product person-hours | Platform engineering person-hours | Reproducible artifact |
|---|---:|---:|---:|---|
| A. Reproduction, preregistration, ethics | 18–30 | 4–8 | 12–22 | signed protocol and baseline replay |
| B. New reader collection & gold labeling | 35–70 | 8–18 | 12–25 | permissions, gold set, annotation agreement |
| C. Controlled feature/augmentation research | 50–95 | 8–16 | 28–60 | experimental registry, ablations, failures |
| D. Independent reader evaluation and statistics | 28–55 | 8–14 | 16–30 | reader-level CI, comparisons and test accountability |
| E. Product feedback validation | 10–22 | 42–80 | 30–55 | calibrated API tests, UX studies |
| F. Research writing/reproduction | 35–70 | 6–12 | 7–15 | research paper, protocol, model card |
| **Total planning range** | **176–342** | **76–148** | **105–207** | none of these are completed work |

The institution determines what formally qualifies as academic research credit; unattended machine runtime does not automatically count as person-hours. A research analysis hour requires a documented task, decisions/observations, output and provenance. Budget hours are not approved academic credits.

## Timesheet schema

`date,person,category,activity,project_goal,hypothesis,pipeline,start,end,duration_hours,artifact_link,review_status`.

Use `category=academic|product|engineering|compute`. For `compute`, record the machine hours independently; do not double count overlapping unattended jobs as human effort. A corrected duration can only follow measured logs.

## Prioritized launch

- S0: audit existing Stage A build versus archived public repository; reproduce on API and freeze manifest/checkpoint versions.
- S1: train P00/P01/P02/P03/P06 *in feasibility first*, profile memory/time; P04/P05 gated by licensing/resources.
- S2: recruit real second same-tradition speaker **before** making any cross-speaker improvement claim.
- S3: only after reader development folds exist, test P08/P09/P10 under correct controls.
- S4: single final test + separated paper/release decisions.

Academic objectives and product KPIs must remain separate in Dvir OS while linking to the same validated evidence.
