# Taamify — CONTINUATION_STATE / سجل מצב מהימן
**Snapshot at:** 2026-10-29 (GitHub connector read; local machine not inspected)  
**State owner:** one orchestrating agent; update only after reading current GitHub/local revisions and actual evidence.  
**Primary source:** `docs/research-os/MASTER_COMPLETION_PLAN_HE.md`.  
**Resume prompt:** `docs/research-os/CONTINUE_AGENT_PROMPT_HE.md`.

## Meaning of statuses

- `verified_complete`: artifact and exact commit / test evidence were inspected.
- `previously_reported`: existing document describes this as done, but not reverified on a fresh environment.
- `in_progress_unconfirmed`: owner says another agent is working; no synced proof of work in current source.
- `candidate_review`: proposal/artifact exists; it has not been released or demonstrated.
- `not_started_visible`: no accessible evidence of completion; **does not assert local work is absent**.
- `blocked`: cannot responsibly proceed until specific rights, data, approval or test evidence exist.

## Remote GitHub inventory at initial snapshot

| Location | What was actually seen |
|---|---|
| `v3-tradition-finding` vs `master` | GitHub reported +104 commits on development branch, with Stage A, Research OS and Task1 documents |
| [Issue #4](https://github.com/MeiRR17/Taamify/issues/4) | open, **zero comments** and no linked completed report visible when checked |
| [PR #5](https://github.com/MeiRR17/Taamify/pull/5) | open/draft research brief with four initial files; +4 commits relative to `v3-tradition-finding` |
| [PR #3](https://github.com/MeiRR17/Taamify/pull/3) | master questionnaire, unanswered GitHub version; user answered A/B separately in chat |
| `docs/research-os/github-study/FINAL_RESEARCH_REPORT.md` | no evidence of published file in examined GH branches; must search local |
| Astra-authored plan | not identified on accessible GitHub branches/search; may exist under a different filename or local checkout |

**No agent progress is certified by this snapshot.** Issue/PR absence does not imply agent inactivity or no uncommitted local work.

## Work items and state

| ID | Status | Grounding / evidence | Exact next action |
|---|---|---|---|
| T00 | `in_progress_unconfirmed` | User reports an agent is working; GitHub has no new report for Issue #4 | inspect local + remote branches, untracked report, Astra plan, publish sanitized state reconciliation |
| T01 | `previously_reported` | `PLAN_STAGE_A.md` and `Data_Prep/stageA_replay.py` in `v3-tradition-finding` | verify same active model, dataset hashes and API replay if needed |
| T02 | `blocked` | Ben13 permissions unresolved in prior audit | request documentary evidence / form rights register; don't assert cleared |
| T03 | `blocked` | no verified new independent reader and authentic errors in remote evidence | design privacy-aware acquisition + labeling; no outreach without approval |
| T04 | `not_started_visible` | PR #5 provides brief README-level prior-art review, not full source comparison | deep dive Technion/Cantillate/TaamimFlow code, claims and controls |
| T05 | `not_started_visible` | 116 discovery leads, many unverified; review matrix not received | evaluate by evidence E0–E4 and license; produce 20 deep reviews |
| T06 | `candidate_review` | five preliminary pilot hypotheses in PR #5 | exact preregistered test plans and accept/reject gates |
| T07 | `candidate_review` | `TASK1_CHANNEL_CALIBRATION.md`; not served | verify candidate B with full E2E/real errors before approval |
| T08 | `not_started_visible` | proposals only; no newly tested code in GH evidence | small licensed isolated feasibility pilots after G0 and rights |
| T09 | `blocked` | cross-reader gold dataset and real wrong reading unavailable in GH evidence | begin only after G2 |
| T10 | `not_started_visible` | research protocol draft, no complete paper in this snapshot | novelty matrix, preregistration and manuscript outline |
| T11 | `previously_reported` | existing webapp; no verified multi-reader practice UX | review current UI with teacher, define narrow pilot |
| T12 | `not_started_visible` | no new integration CI or release report verified | hardening after evidence |
| T13 | `blocked` | G2/G5 and actual student-error tests not met | owner review only when proof supplied |
| T14 | `blocked` | no verified complete paper/venue and G4 evidence | publish manuscript only after review |
| T15 | `not_started_visible` | no verified market interview record | prepare interviews/pricing research; no contact without approval |
| T16 | `not_started_visible` | Dvir OS integration proposal, not verified live | only sanitized read-only adapter after permission review |

## Known baselines (recheck; not fresh results)

- Taam accuracy **85.7%**, AUC **0.993** on stratified 8-parasha Ben13 replay.
- A3 false accusation **2.1%**, target <=2%.
- A4 proxy catch **92%** — not genuine learner error sensitivity.
- Task1 grammar candidate B: combined flagged-correct **13.7% → 2.8%**, proxy-catch falls substantially; *not live*.
- Existing Stage A did not prove cross-reader or commercial readiness.

## Decisions log / approval gates

| ID | Question | Owner status | Blocking |
|---|---|---|---|
| D01 | Rights to use Ben13 source audio for AI training, commercial release, research publication | unresolved | training/release/redistribution based on uncertain rights |
| D02 | Independent Sephardi-Jerusalem reader recruitment and genuine error consent | unresolved | new reader validation |
| D03 | Is Taamify a paper independent of master's thesis? | user preference: separate paper; thesis not selected | academic framing final scope |
| D04 | What is an acceptable performed-taam variant between two correct Jerusalem-Sephardi readers? | user not yet answered | annotation adjudication and false accusation policy |
| D05 | Narrow Beta scope and teacher-reviewed acceptable abstention | pending | public release |
| D06 | Live grammar candidate B threshold change | **not approved** | production modification |
| D07 | Paid compute, major sweep or external audio processing | **not approved** | expensive work / third-party transfer |
| D08 | Journal and article claim/novelty after direct prior-art study | pending | submission |
| D09 | Merge into master or production deployment | **not approved** | publish/deploy |

## First "תמשיך" recommended start

1. Read local git status/branches/untracked files **without modifying**. Determine whether the research agent has produced a result that GitHub does not yet display; inspect and sanitize.
2. If new local research exists: reconcile 116 leads + evidence and flag missing work. Do not discard it or replace with earlier plans.
3. If no research exists: conduct T04 deep-dive on direct prior art and T05 ranked first 20 sources while T02/T03 remain explicitly blocked, produce a verifiable report, and update the state file.
4. Return exact artifact links, commit/test evidence, next 1–3 tasks; ask for approval only if the next safe task is genuinely blocked.

## Update template (for future cycles)

Add to the top of this file after each actual cycle:

```
### Cycle YYYY-MM-DD / <id>
- local and remote branch SHAs:
- agent discoveries / verified outputs (links):
- Tasks moved to verified_complete:
- New facts (source, artifact hash, date):
- Tests run and failures:
- New blocked decisions:
- Next task and acceptance:
- Owner approval requested:
```

**If multiple agents work at once, do not update this shared file in parallel:** orchestrator first checks HEAD, reviews diff, then applies one small mergeable change.
