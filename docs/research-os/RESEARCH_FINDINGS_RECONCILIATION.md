# Taamify — RESEARCH_FINDINGS_RECONCILIATION (2026-10-29)

Every claim found in local or remote work, graded. `accepted` = artifact inspected and/or rerun this
cycle; `needs-validation` = plausible, not reproduced; `rejected` = contradicted by evidence;
`blocked` = cannot be resolved without data/rights/approval.

| # | Claim (source) | Grade | Evidence / reason |
|---|---|---|---|
| R01 | Stage A: 85.7% on 8 parashot, AUC 0.993 (`PLAN_STAGE_A.md`) | needs-validation | not rerun; one reader |
| R02 | A3 false accusation 2.1% (`PLAN_STAGE_A.md`) | needs-validation | `/api/check_phrase` on 160 clean phrases gives 4.0% [2.6, 6.3] "wrong" on correct readings; endpoint/definition differ (F10) → T01 |
| R03 | A4 92% error catch | accepted *as proxy only* | swapped-label negatives, not learner errors |
| R04 | Channel candidate B: 13.7% → 2.8% grammar flags on correct words; 14/111 → 2/111 via API (`CONTINUATION_20261027.md`) | needs-validation | tests pass; replay not rerun this cycle; not served |
| R05 | Train/val leak: 5,817 shared source blocks (`RESEARCH_EVIDENCE_20261027.md` F01) | accepted | consistent with `00_validate_and_prepare_data.py` row-level split; affects model selection, not test |
| R06 | Source-independent split ready, zero cross-split overlap (`INDEPENDENT_DATASET.md`) | accepted | `verify_dataset()` passed 2026-10-29 |
| R07 | Fresh CNN pilot dev macro-F1 0.774 (`pilot_report.json`) | accepted *as a 5-epoch pilot* | not comparable to served model; single seed |
| R08 | Runtime bundle binds served components | accepted | validate → OK |
| R09 | F02 NFD dagesh invariance bug (`webapp/server.py` `expected_channel_label`) | needs-validation | repro snippet in evidence file; not rerun; no fix committed |
| R10 | F03/F04 intake did not fail closed | accepted, fixed in code | `intake_contract.py` + tests |
| R11 | **Technion Jerusalem model 99.0% vs our 89.4% on 40 Ben13 phrases** (`benchmarks/jerusalem_small_20261028`) | **rejected as a capability comparison** | 37/40 phrases were in its training split (exact join, `technion_split_overlap.py`) |
| R12 | On 160 phrases unseen by both: Technion 91.3% vs ours 84.3%, paired +6.8 [+2.4, +11.4] | accepted *same-reader only* | measured this cycle, T04 §1b |
| R13 | Technion models generalise poorly to other readers (F1 0.05–0.52) | needs-validation | their own CSV; reader identities and nusach mix unknown |
| R14 | Tiny Technion model runs locally, 3/3 ta'amim on one clip | accepted *as a smoke test* | `tiny_smoke_20261027.json` |
| R15 | Technion code licence | accepted: **none** | no LICENSE in history; API `license=null` |
| R16 | "116 audited repositories" | rejected | 116 are discovery leads (PR #5); 4 examined in depth here |
| R17 | Cross-reader generalisation of any Taamify model | blocked | G2: no second Sephardic-Jerusalem reader |
| R18 | Learner-error recall | blocked | G2: no genuine labelled errors |
| R19 | Ben13 commercial/training rights | blocked | D01 |
| R20 | Haftara model validity | needs-validation | served from a random split (`local_audit.md`); outside Stage A |
