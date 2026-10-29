# Taamify — תוכנית האב לביצוע, מחקר מדעי, מוצר ומסחור (2026-10-29)

> **סטטוס:** תוכנית ביצוע מותנית בראיות ובאישורים. לא אישור לשינוי live model, הוצאת כספים, פרסום קולות או Merge/Deploy.
> **מקור:** דוחות Taamify הזמינים ב־GitHub בענף `v3-tradition-finding`, מסמכי Research OS, ותוכנית המחקר ב־PR #5. **נכון לבדיקת GitHub ב־29.10.2026, ב־Issue #4 אין דוח/תגובות סוכן וב־PR #5 יש עדיין רק ארבעת קובצי התכנון המקוריים.** אם קיימות עבודות מקומיות חדשות — יש לייבא את *ממצאיהן המסוננים* אחרי בדיקה ולא להמציא אותן.
> **פרומפט הביצוע המתמשך:** `docs/research-os/CONTINUE_AGENT_PROMPT_HE.md`; **מצב משימות:** `docs/research-os/CONTINUATION_STATE.md`.
> **מטרת המשתמש:** בתוך 3 חודשים מערכת שמישה לאנשים אמיתיים אם הבדיקות מאפשרות זאת; סביב 12 חודשים Beta שכבר פועלת ושוק; שלוש שנים השתלבות משמעותית בתשתית לימוד. מסלול אקדמי לפרסום מאמר יישומי עצמאי; **לא הוחלט** ש־Taamify יהיה תזת התואר השני.

## 1. הגדרה: מוצר אחד, שתי דרכי הצלחה

**Product:** פלטפורמת למידת קריאת התורה מבוססת AI; קליטת פסוק/מקטע מוקלט; זיהוי `taam_performed` מתוך האודיו; אימות מול `taam_expected`; משוב מובן, מדויק, בטוח ומתאים למסורת ספרדית־ירושלמית. גרסה ראשונה מוגבלת להחלטות מאומתות. קורא יכול לקבל `correct`, `wrong` רק בראיה אמינה, או `uncertain` כשהמודל לא מסוגל לקבוע. מצב לימוד, פירוק למילים/מקטעים, השוואת מורה והתקדמות לימודית — לפי שערי UX.

**Research:** מחקר יישומי הניתן להגשה בכתב עת, הבוחן האם ייצוגים מוזיקליים/אקוסטיים ולמידת דיבור עוזרים *להכללה לדוברים חדשים* ולזיהוי ביצועים ולא רק לחיזוי הטעם הקאנוני מהטקסט. נדרשים השוואה לעבודה קודמת, ניסויי ביקורת והצהרה אמיתית על מגבלות. נייר מדעי יכול להצליח גם אם תנאי מוצר מסחרי עדיין אינם עוברים; מוצר לא יכול להציג תוצאות מחקר כראיה להוראה בטוחה.

**Business:** מפת משתמשים/משלמים, זכויות ההקלטות/הטקסטים, מחקר שוק, מורה/הורים/מוסדות, פיילוט מוגבל, חוויית שימוש ואומדן עלות — אחרי התמודדות עם reliability ו־licensing. אסור לסמן Product–Market Fit לפי מספרי model accuracy.

**Scope locked for initial baseline:** ספרדי־ירושלמי; טעמי קריאת תורה; ארבעת ערוצי stress/dagesh/guttural/sheva כפי שקיימים אם הוכחו. הפטרות ונוסחים אחרים מחוץ למנוע הראשונה עד למחקר/תיקון split. מורה אדפטיבי מורכב, מסורות נוספות, Kamtz Katan, rare four taamim, scale rollout — backlog נפרד, לא הבטחת v1.

## 2. מקור אמת ומצב הוכחות (לביקורת מחדש לפני כל צעד)

| Claim | State as of 29.10.2026 | Evidence and caveats |
|---|---|---|
| Stage A live API | Reported 85.7% taam accuracy on 8 parashot (40 blocks/parasha), AUC 0.993 | `PLAN_STAGE_A.md`, `Data_Prep/stageA_replay.py`. 8 parashot **same Ben13 reader**; not generalization to new readers |
| False accusation | 2.1% | Above proposed <=2.0% threshold, though sampling uncertainty means no firm estimate of violation size |
| Error detection | 92% | **proxy** wrong expected-taam against correct original audio; not actual student-error sensitivity |
| Grammar channel threshold candidate B | Combined 'wrong' on correct words 13.7% → 2.8%; proxy catch roughly 65% → 38% | `TASK1_CHANNEL_CALIBRATION.md`; candidate not served; not proof of real wrong-reading recall. Some warnings MODERATE rather than demand to repeat |
| Serve mismatch hazards | prior join `verse+block`, CNN/VerifyHead, raw spans, checkpoint parity | tests must run against `webapp/server.py` with actual audio; old 32% join-by-duration corruption documented |
| Haftara | reportedly served model `random_top21_haftmix_FINAL` with random split | **not** product-validated by Stage A; quarantine from evidence-backed claims until split corrected |
| Training | 13 P00–P12 families, A00–A08 augmentations; theoretical 51 fits/1,350 fit-epochs | No proof all needed, runnable, licensed or measured; avoid "train all" |
| GitHub landscape | 116 candidate names in PR #5; preliminary prior-art overview | **not** 116 audited repositories; agent results for task #4 not visible in GitHub on stated check |
| Academic plan | Lecturer encouraged converting former course project to a paper, with no current formal rubric | Journal, contribution and submission deadline undecided; user's *different* master's thesis topic not chosen |
| Data rights | Ben13 rights documentation remains open in prior audit | No commercial training/publishing decisions before permissions verified |

**Read the actual current local checkout before treating this table as fresh**: the agent may have unpushed results or updated thresholds. Classify every report entry as `reported_previous`, `verified_current`, `new_experiment`, or `not_verified`.

## 3. Source-of-truth reconciliation / governance (Stage G0)

1. `git status --porcelain=v1 -b`, `git branch -vv`, `git log --all -n 20 --date=iso`, `git remote -v` (sanitize credentials), compare local vs `origin/v3-tradition-finding`. Fetch safely; never `reset --hard`, `clean -fd`, force push, stash/drop uncommitted changes without owner approval.
2. Check PR #2/#3/#5 and issue #1/#4. Search local and all remote branches for agent report, Astra-authored plan and new artifacts. Record SHA/paths, file age, changed lines and evidence links. **If new research exists, synthesize it first.** Do not overwrite another agent's active branch. When a plan on one branch conflicts with a newer local one, display a decision diff.
3. Look for `results/research_os/audit/local_audit.md` and generated test outputs **locally only**. These paths may be intentionally untracked; report sanitized summaries, not files with private names/paths/recordings.
4. Classify work already complete, partial, blocked, invalidated and not started, with evidence. Use `CONTINUATION_STATE.md` and GitHub Issues to track progress. Do not claim remote visibility into local unstaged work.
5. Freeze current served `model/checkpoint`, `class map`, `verify head`, `calibration file`, `data manifest` fingerprints and thresholds. Run read-only API smoke and update baseline card; never silently adopt previous results as current.
6. Produce `docs/research-os/EXECUTION_DECISIONS.md` with **blocked decisions** and alternatives: source voice rights, independent reader recruitment, grading/annotation authority, academic paper venue, licensing, thresholds. Only owner approves release/compute budget.

**Gate G0:** provenance and current state reconciled; no destructive Git; all secret/raw-audio exclusions passed; next task selected without redoing completed work.

## 4. GitHub ecosystem study (Stage G1 — agent Issue #4)

**Scope:** PR #5: `docs/research-os/GITHUB_ECOSYSTEM_AGENT_BRIEF_HE.md`, `GITHUB_ECOSYSTEM_LONG_LIST.md`, `GITHUB_ECOSYSTEM_PRELIMINARY_ASSESSMENT_HE.md`, `research_os/config/github_ecosystem_candidates.json`. Read latest versions. The 116 are discovery entries; for every claim attach source URL, code/weights/data licenses separately, last maintainer activity, compatibility, architecture integration point, risks, verification stage E0–E4 and decision.

**Search order:**
1. Same-topic prior art: `staviv/cantillation`, `kylemath/cantillate`, `Moriahise/taamimflow_project`, PocketTorah, Sefaria and relevant peer-reviewed work. Build *feature-by-feature* and *scientific-claim-by-claim* comparison: prediction of text-expected taam vs **actual acoustic performance**; named musical tradition; reader-independent evaluation; true wrong-reading feedback; teacher interface; publication novelty.
2. Pitch contours: torchcrepe / PENN / librosa. Alignment: MFA / WhisperX / current aligner. SSL: WavLM / W2v-BERT / S3PRL. Data QC: Label Studio/DVC. Robustness: Audiomentations. UX: Cantillate/WaveSurfer. VC only as conditional research — do not synthesize unconsented voices.
3. Review additional candidate projects up to relevance saturation; target at least 20 detailed reviews and 5 independent pilot designs, **not** automatically 100 installs. E0 may include all leads; E2/E3 are reserved for tests.
4. Save `docs/research-os/github-study/FINAL_RESEARCH_REPORT.md`, `REPO_EVALUATION_MATRIX.csv`, `ARCHITECTURE_DECISION_RECORDS.md`, `PILOT_PROTOCOLS.md`, `ACADEMIC_RESEARCH_SYNTHESIS.md`, `LICENSING_AND_DATA_REGISTER.md`, `PRODUCT_IMPACT_MAP.md`, `IMPLEMENTATION_BACKLOG.md`, `EXECUTIVE_DECISION.md`, and validated `research_os/config/github_ecosystem_review.json`. Use `unknown`, never invented license.
5. Distinguish direct README claims from independently replicated results; **archive flags and last commit aren't performance evidence**. No pretrained model with research-only/noncommercial weights in a commercial deployment without clearance.

**Gate G1:** peer-reviewable, referenced study and a short ranked list of 3–5 **proposed** integrations with falsifiable benefits; explicit rejected options and cost estimates. New work received from agent overrides the original *empty-results status* only with evidence.

## 5. Evidence, consent and gold corpus (Stage G2 — parallel and highest blocking value)

- Audit Ben13 origin, grant/permission, ownership, commercial use, AI training, publication/showcase, deletion rights and minors consent. Written permission, not assumed from a website. Do not store names, personal contacts, private WAV or child voice recordings in GitHub/Drive public files.
- Collect genuinely different readers performing **same Jerusalem-Sephardi tradition**, including age, pitch, vocal style, microphone, room and recording consistency. Initial planning target 8 readers (6 development / 2 untouched final), **not** fixed statistical power. Obtain at least second real reader before claiming cross-reader progress. Any recruitment must be lawful, consensual, guardian-approved where applicable.
- Record **genuine** correct and incorrect attempts (wrong taam, stress, dagesh, guttural, sheva, omissions, hesitation); teacher adjudication must describe acoustic *performed* vs canonical *expected*, not infer performed from manuscript label. For each sample: pseudonymous speaker ID, `source_root_id`, `verse_id`, `word_id`, intended/performed taam, intended/performed grammar, tradition, mic/room class, consent_scope, recording date range, alignment QC, expert adjudication and split.
- Double-annotate stratified subset by independent qualified readers; compute agreement and adjudicate discrepancies. Document how to handle acceptable melodic variation within one tradition.
- Run split leakage assertions: same person, original root, session, clip derivation/VC target reference **cannot** span train/validation/final; verse/phrase overlap controls; hold final sealed until choices frozen. **No audio files in Git**, store securely locally, only hashed/sample-count manifest and aggregate statistics.
- Measure between-reader variance first; revise recruitment target if confidence intervals too wide.

**Gate G2:** documented rights, data card and human gold labels for real wrong readings; usable independent validation cohort and locked untouched final test. If missing, no confirmatory cross-speaker paper claims and no broad student beta.

## 6. Model and experiment plan (Stage G3 — gated by G0 and G2 where needed)

### Baselines & negative controls
- P00 = actual current DeepCNN + VerifyHead + served calibration via real FastAPI end-to-end; preserve as frozen challenger baseline.
- P01 = explicit F0, relative semitone movement, duration, energy; same word/phrase and split.
- P02 = multiresolution and stressed-syllable windows, compare **equal-duration/random window** and crop geometry.
- P03/P04 = frozen WavLM/W2v-BERT embeddings with same lightweight head and source grouping; feasibility-first.
- P06 = CNN+F0 fusion, only if P01 demonstrates information incremental beyond baseline.
- P05 = MuQ music embeddings: **research-only until license confirmed**, separate paper track if necessary.
- P07 = teacher-referenced pair verification on genuine wrong attempts; requires rights-cleared reference.
- P08 = voice-converted training-only controlled augment, compare with pitch shift and human melodic preservation. No synthetic speaker counted as real reader.
- P09/P10 = adversarial/contrastive reader invariance, requiring adequate distinct real development speakers; no premature run.
- P11 = parameter-efficient tuning of successful encoder; P12 = distill to small reliable low-latency student after winner is validated.

### Falsifiable comparisons
- **H0** reproduce old Stage A served pipeline and expose uncertainty;
- **H1** F0/prosody or pitch contour fusion improves on actual performed accent classification vs P00;
- **H2** frozen speech encoders improve generalization on *new readers*, compared at matched splits and seeds;
- **H3** preserving the intended melody under pitch/room/voice augmentation improves real-reader results;
- **H4** teacher-referenced verification better detects genuine reading errors at fixed false-accusation budget;
- **H5** acoustic-only vs text-only vs text+audio and **swapped-verse** controls show the model truly listens;
- **H6** speaker invariant methods improve reader-held-out balanced accuracy;
- **H7** smaller models retain E2E parity, false-warning safety and latency.

Use research protocol `RESEARCH_PROTOCOL.md` H0–H7 as canonical for preregistration; the above H1–H7 are **conceptual workstreams, not a replacement numbering** — map to preexisting hypothesis IDs before execution; don't silently renumber protocols.

### Experiment protocol per candidate
`experiment_id, hypothesis_id, source_project_SHA, local_code_SHA, dataset_manifest_SHA, root_split_SHA, config_SHA, checkpoint_SHA, calibration_SHA, random_seeds, device, observed_memory, feature_extraction_wallclock, actual_minutes_per_epoch, outcome, error_analysis, stop_reason, review_status`.

Primary scientific metric: reader-level **balanced accuracy** on genuine new readers with paired comparison to P00, CIs clustered by reader and predefined practical effect size. Secondary: per-taam F1/recall, per-phrase consistency, confusion hierarchy, calibration, cost and abstention. Primary product: **true-error recall at fixed false accusation <=2% on actual correct readings**, with coverage and latency. Single speaker and thousands of correlated words ≠ thousands of independent people.

Controls: shuffled labels; expected text alone; swapped audio while retaining verse; uniform-pitch/contour attacks; same-width random crops; exact same-word different performances; model-prediction pseudo-label vs independent human adjudication; speaker probe in embeddings; train-only variants of source-root; frozen vs tuned encoders. Pre-register threshold selection, statistical comparisons/multiplicity, seeds and final test access. At least 3 independent seeds for finalist training; **do not repeat final holdout tuning**.

**Gate G3:** finalist beats matched baseline by practical and statistical evidence without cheating the false-accusation budget, or negative result is documented. No expensive full 51-fit schedule by default.

## 7. Applied paper track (Stage G4 — independent review)

**Research question** should be narrow and defensible: e.g. "Do acoustically grounded, reader-invariant representations support accurate evaluation of performed Biblical cantillation in Jerusalem-Sephardi Torah reading, beyond text-only prior predictions?" Could instead focus on data/benchmark and error calibration if that is the actual contribution. Decide after prior-art review; do not manufacture theoretical novelty.

Paper deliverables:
- Title, abstract, problem definition, literature review including 2025 Technion and other Torah-cantillation work; ethical corpus acquisition, data dictionary; acoustic-vs-text leakage controls; preregistered experiment matrix.
- Baseline exact SHA, split protocol, model variants, calibration, adversarial controls, reader-level results + confidence, per-class and demographic/recording variance where legitimate, ablations, compute and negative results.
- Discussion distinguishing canonical textual accents and performed acoustic patterns; what "correct reading" means within the tradition, limits of expert agreement, why new reader reliability remains open or demonstrated.
- Reproducibility pack without private recordings: scripts, sanitized schemas, citations, version lock, aggregate tables and license register.
- Venue shortlist from latest verified scope/requirements; e.g., compare digital humanities / music information retrieval / speech processing outlets but verify dates/requirements/fees/ethics, not automatically promise Q-ranking, impact or acceptance. No requirement to attach this paper to existing master's thesis.

**Gate G4:** scientifically defensible contribution + independent measurements + reviewer-quality manuscript and references; target venue separately approved by owner.

## 8. Product, privacy, UX and release (Stage G5 — may proceed parallel to G3)

Journey: student chooses parasha/aliyah → hears teacher example → reads word/phrase/verse → system aligns with confidence → distinguishes actual from expected → gives exact correction or abstains → offers local replay and practice, tracking progression only with informed consent. Strict Hebrew RTL accessibility and clear feedback; prioritize correct negative feedback over gamification.

Product acceptance proposals (must be approved and validated, not status claims):
- False accusation on naturally correct readings **<=2%** on development-chosen threshold and independent evaluation.
- Real wrong-reading recall **>=70%** under same threshold, reported with class/reader CIs.
- Coverage and abstention tradeoff reported; initial planning preference <=20% abstention but must not force risky judgments; live user testing with instructor review.
- End-to-end latencies p50/p95 measured on real target phone/network/server; error localization, frontend recording errors, audio mismatches, 4xx/5xx/rollback and user data deletion.
- Checkpoint/verify-head/class index and calibration fingerprint assertions at startup; pipeline e2e regression after every model swap.
- Mobile UI teacher reference and robust browser permission state; students must be able to say "feedback is wrong."
- Rights/consent, age-sensitive privacy, versioned terms, data retention and minimization, redaction from logs. No app public beta without owner release approval.

**Beta scope decision:** Torah-only one named tradition; a constrained private expert pilot if independent reader quality is insufficient; public beta only after gates. Haftara and multi-tradition postponed until independently retrained and tested.

## 9. Engineering and operational plan (Stage G6)

- Keep current Python/PyTorch/FastAPI + queue unless ADR proves migration necessary; no new orchestration stack merely to adopt one.
- Artifact registry includes immutable code, data, model, calibration, label schema, API version, license and dependency provenance. Status transitions: `planned → feasibility → development_passed → independent_test_passed → approved_release → serving` with rollback.
- M1 Max 64 GB unified RAM primary accelerator; M4 Air 16 GB for CPU/light prototyping; Windows VRAM unknown until verified. Stop idle Ollama models *only if* safe and unused, not `kill -9`; validate free memory, peak memory and queue starvation. Record real wall time/energy if available. **No invented machine hours** from 1,350 planned fit-epochs.
- Data checks: no source-root/speaker across splits, no duration-based file joins, no corrupted F0 due to augmentation, no closed-source/restricted weight export; secret scanners before push.
- Reuse `Dvir OS` only for metadata/status dashboard after access/security review, no voice blobs or automation that manipulates deployed model without approval.
- CI test plan: static schema, unit, property-based key correctness, serialized fake-audio E2E integration and smoke with real locally accessible rights-cleared audio, performance baseline and release gate. Provide runbook commands actually tested, not guessed.

## 10. Business and market validation (Stage G7)

Interview people who actually teach/read Torah, bar-mitzvah students/parents and institutions — not just use personal assumptions. Record `role / pain / frequency / current alternative / willingness-to-pay / discovery channel / consent` without exposing identities. Competitive feature matrix to Cantillate/TropeTrainer/teachers; unit economics (inference cost, storage, human review, rights acquisition), priced alternatives, onboarding friction. Trial conversion and retention metrics require actual pilot data. No claims of adoption, revenues or PMF until measured.

## 11. Dependency-first 90-day scenario (conditional, not promised)

| Period | Main deliverables | Gate |
|---|---|---|
| Days 1–7 | G0 state audit, current agent handoff, citations and evidence ledger; identify right academic/product claim; outreach plan for new voices/permissions | Current state, no unverified data claims |
| Days 8–21 | 116-candidate triage → 20 reviews, rights audit, genuine reader collection begins, annotation rubric, low-cost F0/alignment feasibility, UX baseline | G1 study + G2 recruitment progress |
| Days 22–42 | Gold mini-set, independent speaker pilot, controlled matched 1-seed P00/P01/P03, API regression, candidate calibration and stop/go decisions | G2 prerequisites + G3 feasibility |
| Days 43–63 | 3-seed finalist experiments **if eligible**; expert review of real errors; live latency/device tests, private supervised pilot, academic contribution decision | G3 reproducibility, preliminary G5 |
| Days 64–90 | Freeze candidate, independent final assessment **once only** after protocol lock, conditional beta, paper first full draft + reproducibility appendix, safety+privacy review and Dvir OS status adapter proposal | G4/G5 review approvals |

These are *target windows*. If new real readers, permissions, audio labels or academic novelty cannot be verified, change deliverable to "documented pilot blocker and research report", not fabricate passing results.

## 12. Detailed execution backlog and acceptance

| ID | Next action | Dependency | Acceptance evidence | Priority |
|---|---|---|---|---|
| T00 | Locate agent results locally, PRs and issue #4; compare SHA/diff, avoid overwrite | none | `CURRENT_STATE_SNAPSHOT.md` with source URLs/SHA + pending work | P0 |
| T01 | Validate current served API and Stage A on matched ID strata | T00 | E2E output, served model/calibration hashes, caveats | P0 |
| T02 | Rights audit of Ben13, external voices, assets and model weights | T00 | explicit permissions matrix + blocked entries; no raw data in Git | P0 |
| T03 | Gold dataset plan + recruit real same-tradition readers and actual errors | T02 | consent and human annotation plan; independent labels | P0 |
| T04 | Direct competitor/prior art deep dive (Technion, Cantillate, TaamimFlow) | T00 | claim-level differentiation + evidence citations | P0 |
| T05 | Triage and review 116 sources and beyond | T00 | exact URLs, dates, license, rankings and rejects | P0 |
| T06 | Produce top 5 feasibility protocols and minimal replacement/adapters ADR | T04,T05 | paired baseline, target test, budget and stop criteria | P0 |
| T07 | Verify Task1 grammar candidate B without serving it | T01 | same-split before/after + coverage/proxy distinction + rollback plan | P1 |
| T08 | F0/aligner/SSL first pilots and timing/memory measurement | T02,T06 | reproducible sandbox tests; no final holdout exposure | P1 |
| T09 | Matched cross-reader study and true-error evaluation | T03,T08 | real per-reader CIs, absolute vs conditional metrics | P0 blocked pending data |
| T10 | Publication protocol/preregistration and manuscript skeleton | T04,T06,T03 | novelty table, hypothesis freeze and ethical sections | P1 |
| T11 | RTL practice UX and recorded lesson loop | T01,T02 | usability feedback+latency in private demo | P1 |
| T12 | Integration and security hardening | T07,T09,T11 | E2E, secrets scan, rollback and thresholds | P1 |
| T13 | Product beta acceptance review | T09,T12 | threshold/rights acceptance, explicit owner sign-off | P0 blocked pending data |
| T14 | Academic evidence lock and initial paper submission | T09,T10 | reproducibility pack and validated claims; venue approved | P2 |
| T15 | Market validation and business model pilot | T11,T02 | documented interviews and unit economics; no guessed revenue | P2 |
| T16 | Dvir OS read-only experiment metadata link | T00,T12 | approved interface, permission and redaction tests | P2 |

If another agent has finished a task: mark `verified_complete` **only** with artifact/ref/test; otherwise `reported_unverified` and inspect. Maintain negative results.

## 13. Roles, status reporting and approval

One orchestrator manages the dependency graph and state file. Parallel roles can research prior art; data provenance/ethics; acoustic/audio; model experiments; product/UX; academic paper; security/ops. Parallel independent read-only research is fine. **One owner of split definitions, one immutable final test, one reviewed source of truth**. No branch may overwrite another agent's active work.

At the end of each "תמשיך" cycle:
`[Snapshot ref/sha] [What was reviewed] [What changed] [Evidence and tests] [Quality gates] [Blocked/unknown] [Next 1–3 issue-sized actions] [Approval requests]`.
Do not say "done" for research if only a plan was published. Keep user-visible summary concise but evidence-backed.

**Actions requiring explicit fresh approval:** merging into `master` or protected branches, changing served thresholds/checkpoints, deploying or public beta, heavy paid or unattended training beyond approved budget, external audio/data upload, third-party access grants, recruitment contact on owner's behalf, and publication or submission in owner's name. A command "תמשיך" permits authorized **read-only investigation and non-production planning/test artifacts** on a separate branch, not these gated actions.

## 14. Supporting canonical documents

- Repo source state `https://github.com/MeiRR17/Taamify/tree/v3-tradition-finding`
- `PLAN_STAGE_A.md`, `docs/research-os/TASK1_CHANNEL_CALIBRATION.md`
- `docs/research-os/RESEARCH_PROTOCOL.md`, `PRODUCT_ACCEPTANCE.md`, `TRAINING_HOURS.md`, `ROADMAP.md`
- [Research assignment Issue #4](https://github.com/MeiRR17/Taamify/issues/4)
- [GitHub ecosystem research PR #5](https://github.com/MeiRR17/Taamify/pull/5)
- [Discovery questionnaire PR #3](https://github.com/MeiRR17/Taamify/pull/3)
- `docs/research-os/CONTINUE_AGENT_PROMPT_HE.md` and `CONTINUATION_STATE.md` on the branch of this plan.

**The technical research report will supersede only the parts it actually measures; it does not automatically supersede the author's goals, safety gates, user data permissions or academic rules.**
