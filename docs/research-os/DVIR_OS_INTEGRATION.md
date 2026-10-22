# Taamify Research OS — ארכיטקטורת מחקר מבוססת ראיות

**סטטוס:** מפרט ל־PR סקירה בלבד. לא בוצעה הפעלה של מחשבי הבית, אימון מודלים או קישור חי ל־Hermes. `Dvir OS` נשאר מרכז הפיקוד; אין לשכפל אליו את מנוע ה־ML.

## A. מקורות שנבדקו, ואמת תפעולית

1. `MeiRR17/dvir-os`: `AGENTS.md`, `docs/architecture/overview.md`, `docs/product/requirements.md`, `app/page.tsx`, `docs/sync/README.md`, `config/drive-documents.json`.
2. Google Drive: **Dvir OS — מאיר וציון — אפיון ויכולות v0.9**; **Taamify/PLAN_STAGE_A.md** (עדכון 02.10.2026); **Taamify/CLAUDE.md**.
3. `MeiRR17/Taamify` public `master` is **older** than the local/Drive production notes; it must NOT be used to overwrite the live service. The published `webapp/server.py` still documents the earlier top-11 setup.
4. Stage A source claims 87.9% recognition and 89.4% true-error capture at <=2% false accusations, evaluated on a single Ben13 voice via actual API. This is *documented evidence*, not independently rerun.
5. Existing Taamify discoveries include a file-to-block mismatch fixed (previously 32% mismatches), data leakage from earlier random splits, production checkpoint mismatches, partial grammar-channel miscalibration; 7-model ensemble did not improve performance. **Do not repeat these known failures without a new hypothesis.**
6. The original research environment uses a Ben13-only Jerusalem-Sephardi corpus and a local queue; other traditions must not silently enter the study.

## B. Runtime and resource architecture

```
Hebrew mobile/desktop → Dvir OS Next.js PWA [מאיר | ציון]
                         │       ├─ /research/taamify (read-only study view)
                         │       └─ evidence / approvals (planned)
                         ▼
               Authenticated control plane (NOT YET LIVE)
                         │
              Read-only status adapter / schema v1
                         │         [no raw WAV / secrets]
                         ▼
      MacBook Pro M1 Max / 64GB  — local Taamify worker
      audio manifest → immutable split policy → quality+alignment
                                  ├─ cached features → model registry
                                  ├─ augmentation registry
                                  ├─ scheduler (one heavy GPU job at first)
                                  └─ evaluation / checkpoint registry
                         │
         sanitized run summaries + hashes + model cards
                         ▼
        owner-reviewed release candidate (manual approval)
                         │
             Taamify FastAPI product API + E2E probes
```

**Secondary device:** M4 Air / 16GB: development, smoke tests, lightweight preprocessing. **Windows Ryzen 5 3600 / 32GB:** GPU model and VRAM not verified; GPU train jobs blocked. No guaranteed 24/7 uptime; local notebook can sleep or reboot. Do not open public ports; private authenticated access only. No new pay-per-token/cloud expenses by default.

**Source of truth:**
- Taamify code, experiment configs, model metadata and test scripts → GitHub (appropriate Taamify branch).
- Dvir OS dashboard and orchestration adapters → this repository.
- Canonical planning docs → Google Drive; `docs/drive-mirror/*` is generated read-only.
- Raw audio, learner recordings, feature matrices, licensed speaker exemplars, checkpoints → local controlled storage, never in Dvir OS or GitHub. Only derived, permission-safe summary values flow to control plane.

## C. Component contracts (proposed; adapter not implemented yet)

`taamify-status.v1.json` from an **authenticated local agent**, containing ONLY:
```json
{
  "schema_version": 1,
  "source": "taamify-local",
  "captured_at": "ISO8601",
  "project_revision": "commit SHA",
  "dataset_revision": "hash of allowlisted manifest",
  "serving_checkpoint_revision": "opaque short ID",
  "host_id": "mac-pro-m1-max",
  "jobs": [{"run_id":"opaque", "pipeline_id":"P03","state":"queued|running|passed|failed","started_at":"ISO8601|null","elapsed_seconds":0}],
  "measurements": [{"run_id":"opaque","split_id":"pre-registered","metric":"balanced_accuracy","value":0.0,"n_speakers":0}],
  "attested": false
}
```
Never include paths containing usernames, emails, IPs, authentication tokens, raw transcripts, voice files, personal speaker identities, secrets or private third-party data. Host only via an authenticated owner endpoint, with request size limits, timeouts and read-only credentials. The current dashboard uses a checked-in **documented baseline snapshot**, not this contract.

**Required integration layers:** (1) lightweight exporter in Taamify's real local checkout, (2) local-only heartbeat + watchdog (Hermes not assumed connected), (3) private read-only authenticated Dvir OS status endpoint, (4) dashboard view and audit trail, (5) explicit review-before-release action. Do not implement automatic production model swaps or job launching via unauthenticated UI.

## D. Data and research workflow

Raw authorized corpus → immutable manifest `audio_id, speaker_id_pseudonymous, verse_id, clip_id, taam, tradition, consent_scope, audio_hash, source_root, synthetic_parent_id` → QC/forced alignment → gold manual audit → speaker+root-aware frozen splits → train-only augmentation → cached acoustic/prosody/SSL features → model training → validation and tuning → **one final sealed speaker test** → interpretation → separately gated product API evaluation.

Key constraints:
- Keep Sephardi-Jerusalem single-tradition scope unless a *new, separate* protocol is approved.
- Synthetic variants inherit the **same source root** as the original audio; speaker references for voice conversion must be strictly from training-approved data and may not come from held-out speakers.
- A reviewer must independently label real errors, correct readings and ambiguous recordings. Human annotations and inter-annotator agreement belong to research evidence.
- Gold-set controls: same-word/different-voice, same-speaker/different-taam, shuffled labels, same-width random windows, realistic microphone/room, F0-preservation tests, and existing pitch-shift baseline.
- Context-aware sequence model is measured separately: it may 'correct' a student's actual error by relying on textual expectations.
- Frozen train/validation/test split is by **speaker AND recording root**, not random individual words. Tune only on development speakers; do not repeatedly peek at the sealed final speaker test.

## E. Experiment engine

Catalog source: `config/taamify-research.json` (P00–P12) records hypothesis, dependencies, fit count, epochs, status, resource targets. Actual Taamify worker should keep its existing proven job queue, with a machine-aware adapter, not a competing replacement. Artifacts: run config digest, seed, package lock/hash, code SHA, manifest/split digest, per-speaker metrics, calibration, error analysis, CPU/RAM/GPU peak, wall time and energy estimate if measured, user review status, model card.

**Stages:** 0: provenance and E2E baseline audit; 1: 1-seed pilot + memory feasibility; 2: replicated controlled development with 3 seeds and reader folds; 3: locked final test on fresh readers after model freeze; 4: science report + separate release gate. Paired comparisons only on exactly the same examples.

## F. Release gate (independent of research claim)

- Versioned active serving model/checkpoint, VerifyHead compatibility assertion, fixed class IDs and feature params.
- Actual client WAV → FastAPI → user-facing feedback full E2E; parity with stored processing and ONNX if used.
- Error-detection sensitivity at a calibration-selected false accusation rate; report uncertain/abstain coverage and failures across people and devices.
- Manual sign-off before exposing to learners, especially minors; data minimization and speaker consent.
- Dashboard may show a verified research **result**, but cannot label it production-grade until separate release evidence exists.

## G. Responsible publication

Record hypotheses, preprocessing, excluded classes, negative findings, licenses, reproducibility and limitations. Avoid asserting academic credit for unattended GPU hours. Publication-ready contributions require independent speakers and good-faith reproducible evaluation.

## Literature anchors (research avenues, not proven Taamify gains)

- Chen et al., WavLM: https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/
- Meta W2v-BERT 2.0 model card: https://huggingface.co/facebook/w2v-bert-2.0
- Seed-VC upstream (archived Nov 2025): https://github.com/Plachtaa/seed-vc ; only authorized voices, assess licensing and melody preservation.
