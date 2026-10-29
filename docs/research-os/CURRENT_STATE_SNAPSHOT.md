# Taamify — CURRENT_STATE_SNAPSHOT (T00)

**Taken:** 2026-10-29, local M1 Max checkout + GitHub API. Supersedes the "local machine not inspected"
caveat in `CONTINUATION_STATE.md`'s initial snapshot.

## Git

| Ref | SHA | State |
|---|---|---|
| `origin/v3-tradition-finding` | `78ca11d` | code mainline; unchanged this cycle |
| `codex/channel-calibration-review-20261027` | `78ca11d` → `a31be35`, `a7b99cc`, `687a3f7` | **was local-only with ~40 uncommitted files** (27–28.10 agent work); now committed and pushed |
| `docs/agent-continue-masterplan-20261029` (PR #6) | `e6124e6` | plan + ledger; base of this cycle's docs branch |
| `docs/continue-cycle-20261029` | this branch | snapshot, reconciliation, T04 report, ledger update |
| `docs/github-ecosystem-landscape-20261028` (PR #5) | `ef9c98f` | 116 leads, unchanged |
| `master` | `d090dfa` | initial commit only; not touched |

Issue #4: open, one owner comment (2026-10-29), no agent report before this cycle.
No Astra-authored plan was found in any local or remote branch, in `docs/`, or under `results/`.

## What the unpushed local work contained (now on `codex/channel-calibration-review-20261027`)

| Area | Files | Verification this cycle |
|---|---|---|
| Channel-calibration candidate wiring (Task 1 follow-up) | `webapp/channel_calibration.py`, `server.py`, `practice.html`, `prepare/replay_channel_candidate.py` | tests pass; candidate **not served** (env var only) |
| Runtime bundle (SHA binding of every served component) | `webapp/runtime_bundle.py`, `research_os/model_bundle.py` | `model_bundle validate observed_20261028.json` → OK |
| Human-reviewed intake contract (fail-closed) | `Data_Prep/intake_contract.py`, `voice_intake.py`, `ingest_collected_phrases.py`, `webapp/reading_contract.py` | tests pass; no recording ingested |
| Source-independent Ben13 split | `research_os/prepare_independent_dataset.py` | `verify_dataset()` → ready, fingerprint `6dfa79cb…` |
| Bounded dev CNN pilot on that split | `research_os/train_independent_candidate.py` | report read: 5 epochs, dev macro-F1 0.774, acc 0.777; test/calibration never decoded |
| External-model tools (Technion Whisper) | `try_external_cantillation.py`, `benchmark_candidates.py` | reused this cycle (see T04) |
| Diagnostic pack (blind human review page + phrase benchmark) | `build_diagnostic_pack.py`, `source_audio.py` | tests pass; 200 review units built, **0 annotated** |
| Readiness audit (spectrogram-exact split check) | `audit_research_readiness.py` | report read: train/val share 5,817 source blocks; train/test 0 |
| Docs | `RESEARCH_EVIDENCE_20261027.md`, `MODEL_REUSE_REVIEW_HE.md`, `MASTER_PLAN_HE_20261027.md`, `INTAKE_WORKFLOW_HE.md`, `RUNTIME_BUNDLE_HE.md`, `CONTINUATION_20261027.md`, `INDEPENDENT_DATASET.md` | privacy/secret scan clean |

Tests executed 2026-10-29: `venv/bin/python -m unittest research_os.test_*` → **91/91 OK** (90 existing + 1
new); `node research_os/test_practice_channels.js` → passed. Secret scan (tokens, keys, absolute paths,
emails, audio/weights extensions) on every committed file → no hits. `results/`, `prepared_data/`,
`models/`, `app_assets/collected_phrases/` remain gitignored; nothing from them was committed.

## Served state (read-only)

The bundle `results/research_os/runtime/observed_20261028.json` (digest `d9ed41a2…`, status
`observed_binding_not_training_provenance`) validates against the current checkout. Serving was not
changed. The live app on :8000 was not running during this cycle; test servers used :8078 only.

## Not found / not done

- No human annotations exist for the diagnostic pack; no second Sephardic-Jerusalem reader.
- No Ben13 rights documentation (D01).
- `docs/research-os/github-study/FINAL_RESEARCH_REPORT.md` and the other Issue #4 deliverables: not
  produced; this cycle delivered the direct-prior-art part of T04 only.
