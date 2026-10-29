# Workflow — From Taamify discovery answers to a final research/product plan

**Status:** questionnaire prepared for human answers. No final scope, academic claims, training budget or production changes have been approved by the questionnaire alone.

## Source of truth

- Questions: `docs/research-os/MASTER_DISCOVERY_QUESTIONNAIRE_HE.md` (192 numbered questions, 19 sections).
- Scientific baseline: `docs/research-os/RESEARCH_PROTOCOL.md`, `PLAN_STAGE_A.md`, `Data_Prep/stageA_replay.py` and subsequent local audit.
- Agent task blueprint: `docs/research-os/AGENT_HANDOFF.md`; Dvir OS is a review/dashboard control plane, not the ML training repository.
- The lecturer's *exact written instructions and deadlines* are currently **not verified** here. Do not infer them from assumptions about degrees, papers or the model results.

## Completing the questionnaire

1. User completes the Word/Markdown questionnaire locally or returns the filled content in chat. Questions tagged `[ליבה]` (67) are a first-pass priority; unanswered expansion questions remain explicit.
2. Use these statuses for unresolved answers: `לא ידוע`, `דורש החלטת מרצה`, `לא רלוונטי`.
3. Maintain a private answer document containing any identifiable speaker/student information, unpublished budget or device access details. Never commit passwords/tokens, minors' data, voice recordings or identifying student records.
4. Before publishing a response summary to GitHub, generate a **sanitized decision ledger** that omits sensitive data and shows each decision's source ID; obtain owner approval.
5. For any unresolved academic/supervisor question, **do not invent** a requirement. Record a blocking decision and a proposed question to send to the lecturer.

## Synthesis deliverables (after answers)

- `PROJECT_CHARTER.md`: success criteria, boundaries, stakeholders, milestones, academic vs product.
- `ACADEMIC_RESEARCH_PROPOSAL.md`: problem, hypotheses, novelty, background, methodology and literature review plan.
- `GOLD_DATASET_PROTOCOL.md`: independent voices, consent, human annotations, real learner errors, sealed holdout.
- `EXPERIMENT_MATRIX.md`: selection of P00–P12 + A00–A08, control conditions, primary/secondary outcomes, seed policy, leakage guards, failure/stop criteria and negative results reporting.
- `REPRODUCIBILITY.md`: code/data/checkpoint hashes, evaluation scripts, API parity and reproducible run manifests.
- `RESOURCE_PLAN.md`: **measured** minutes/epoch, memory, machine feasibility, person-hours, calendar constraints and budget.
- `PRODUCT_PRD.md` + `ACCEPTANCE_GATES.md`: learners/teachers, UX, false accusation and real-error recall, privacy, latency, versioning and deployment.
- `DVIR_OS_INTEGRATION_PLAN.md`: authenticated read-only status first, review trail, later opt-in scheduling and agent permissions.
- `EXECUTION_ROADMAP.md`: task dependencies, accountable owners, deliverables and review gates.
- `PUBLICATION_PLAN.md`: paper/thesis/course deliverables based on actual lecturer requirements, not guessed.

## Scientific non-negotiables

- Stage A across 8 parashot was reported as 85.7% taam accuracy, AUC 0.993, A3 false accusation 2.1% against a 2% target, and A4 92% **proxy** error catch, not recall on real wrong readings. Record these distinctions; never quietly mark all gates passed.
- Speaker-level generalization is unverified until genuine new Sephardi-Jerusalem readers appear in properly isolated evaluation.
- Synthetic voice variations are not independent participants and cannot cross source/speaker train-test leakage boundaries.
- Research success and product release each require their **own** approval gate.
- Hardware estimates are not measured accelerator hours; academic person-hours and GPU time are separate.

## Acceptance for questionnaire stage

- [x] All questions have stable IDs and explicit answer fields.
- [x] Existing facts distinguished from unknown lecturer requirements.
- [x] Sensitive inputs kept out of GitHub by design.
- [ ] Lecturer requirements received and verified.
- [ ] User answers supplied and gaps reviewed.
- [ ] Scientific protocol and final architecture approved.
- [ ] Agent implementation tickets created from approved scope.

**No agent should begin expensive new research training solely because this questionnaire exists.** Continue the separately authorized calibration/reproduction work while the academic scope is established.
