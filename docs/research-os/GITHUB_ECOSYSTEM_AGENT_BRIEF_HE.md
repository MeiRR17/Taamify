# Taamify — כתב מינוי לסוכן: מחקר GitHub מקיף ליצירת תוכנית פרויקט סופית
**תאריך:** 2026-10-28  
**סטטוס:** משימת מחקר/אבחון, ללא הרשאת אימון, אינטגרציה, התקנות, push למערכת הפעילה, שינויי ספים או פריסה.  
**עדיפות:** מחקר יישומי שמשרת מוצר אחד — מורה ללימוד קריאת התורה בטעמים בנוסח ספרדי־ירושלמי — ומפיק מאמר מדעי שניתן להגן עליו.

## 0. משימה חד־משמעית

אתר, סנן, קרא ובחן פרויקטי GitHub, ספריות, מודלים, קורפוסים, datasets, מאמרים ומוצרים דומים, ככל שיידרש. **אין מגבלת 100 מועמדים**: מותר לנתח 30, 100 או יותר; אין למלא מכסה על ידי הוספת פרויקטים לא רלוונטיים. המטרה אינה רשימה אלא `FINAL_RESEARCH_REPORT.md` מלא, עם ראיות, ניסויי feasibility, דירוגים ותוכנית ביצוע מתוקצבת; וכן קטלוג מכונה `github_ecosystem_review.json`.

תחילה בדוק את `docs/research-os/GITHUB_ECOSYSTEM_LONG_LIST.md` — רשימת מועמדים ראשונית בלבד. **כל רשומה ללא אימות מחייבת בדיקת ריפוזיטורי ולא די בקיום שם נתיב.** חפש מעבר לרשימה: פרויקטים דומים של Cantillation / Torah trope tutor / Biblical Hebrew alignment; מחקרי audio-guided reading / singing assessment; pitch contours, self-supervised speech, forced alignment, controlled voice conversion, student feedback, metric learning, dataset curation, MLOps, deployment on Apple Silicon.

## 1. מקור אמת: קרא לפני המחקר

- `CLAUDE.md`, `PLAN_STAGE_A.md`, `webapp/server.py`, `Data_Prep/stageA_replay.py` ו־`docs/research-os/TASK1_CHANNEL_CALIBRATION.md`.
- `docs/research-os/RESEARCH_PROTOCOL.md`, `PRODUCT_ACCEPTANCE.md`, `TRAINING_HOURS.md`, `AUGMENTATION_PROTOCOL.md`, `ROADMAP.md`, `AGENT_HANDOFF.md`.
- `research_os/config/pipelines.json` (P00–P12) ו־`research_os/config/augmentations.json` (A00–A08).
- `docs/research-os/MASTER_DISCOVERY_QUESTIONNAIRE_HE.md` רק אם זמין ב־branch: תשובות המשתמש ממשיכות להיאסף. אין להמציא החלטות חסרות.
- `MeiRR17/dvir-os` מספק תצוגת dashboard/control plane, לא repository שבו מממשים מחדש את מודלי Taamify.
- אחרי כל read: כתוב commit SHA, dataset/checkpoint provenance, תאריך, נתיב, האם האמור נמדד, מתוכנן או השערה.

## 2. מה כבר ידוע (לא לשכתב כאילו נמדד מחדש)

1. Stage A של מודל קול אחד: replay מדגמי על 8 פרשות מדווח 85.7% דיוק, AUC=0.993, **2.1%** false accusation מול יעד <=2%, ו־92% זיהוי **proxy errors**. האחרון אינו מדד לשגיאות מבוצעות אמיתיות.
2. `TASK1_CHANNEL_CALIBRATION.md`: מועמד בלתי מוגש מפחית false flags משולבים ברמת מילה מ־13.7% ל־2.8%, במחיר ירידה בפרוקסי catch מכ־65% לכ־38%. אין אישור החלפת live threshold.
3. בעיית `real_block_clips` שהתאימה בלוקים לפי משך הייתה בעבר טעות: להשתמש ב־verse + block מזהים מדויקים. baseline replay דורש API אמיתי, לא רק specgram ממוין.
4. קוראים נוספים **באותו נוסח** + קריאות שגויות **אמיתיות** הם חסמים; אף מספר augmentations אינו תחליף.
5. Windows GPU/VRAM עדיין לא מאומת; Mac M1 Max 64GB הוא GPU MPS העיקרי. לפני כל benchmark אמיתי לוודא משאבים, Ollama ורישיונות.
6. המשתמש רוצה מוצר שמיש במהרה (יעד חזון 3 חודשים) ו־Beta/שוק בטווח שנה; אלה מטרות לא התחייבות טכנית. מטרת המאמר היא מדע יישומי, לא סיפור שנכתב אחרי בחירת המודל.

## 3. מסלולי חיפוש — כולם חובה

R1 **Direct prior art:** Torah trop, Cantillation, Hebrew biblical ASR, student pronunciation evaluation, cantillation-synthesis/feedback. תעד חפיפה מדויקת וזכויות.  
R2 **Audio integrity & alignment:** מילה/הברה, VAD, CTC/MFA/WhisperX, canonical text vs performed text, same-verse matching, device SNR.  
R3 **Acoustic & melody:** F0, semitone contours, mel/STFT, temporal envelopes, duration/stress, pitch DTW/sequence edit, non-melodic confound controls.  
R4 **Pretrained speech & music:** WavLM, HuBERT, W2v-BERT, music encoders, frozen probes, fine-tune/LoRA, distillation, model weight licensing.  
R5 **Synthetic augmentation/VC:** RIR, noise, codec, time stretch, pitch shift, formants, voice conversion with contour-preservation; archived projects and license risks.  
R6 **Reader generalization:** speaker adversarial/contrastive learning, group CV, speaker privacy and source-root leakage, true unseen reader holdouts.  
R7 **Verification & pedagogy:** intended/actual taam, correct/incorrect genuine utterances, uncertainty/abstention, "לא בטוח", teacher gold, actionable error localization.  
R8 **Corpus & Hebrew:** Sefaria, MAM/OSHB, Hebrew morph/nikud/taamim, parasha/aliyah/haftara, orthographic normalization, copyright/attribution.  
R9 **Annotation/MLOps:** gold set tools, quality review, MLflow/DVC, seeds/hash provenance, track all failures.  
R10 **UX & deployment:** RTL real-time recording in mobile browser, feedback latency, WebAudio, FastAPI, M1 Max MPS, ONNX/CoreML parity.  
R11 **Privacy/security/quality:** audio recording rights, consent for minors and VC, secrets, dependency scanning, license inventory, model-card/data-card.  
R12 **Related scholarship:** find original papers/baselines of top repositories, replicability and citation metadata. Distinguish GitHub quality from peer-reviewed evidence.

## 4. חובה לבצע עבור *כל* candidate (גם מועמד שנדחה)

אחסן את הרשומה ב־JSON/CSV + טבלה; שדות:
`id, lane, repo_full_name, canonical_url, owner, description, query_that_found_it, evidence_readme_url, evidence_license_url, evidence_release_url, checked_at_utc, latest_commit_date, latest_release, archived, stars_optional, open_issues_trend, language, python_version, dependencies, license_code, license_weights, license_training_data, commercial_use_verdict, third_party_notices, citations, similar_taamify_component, unique_capability, proposed_change_in_taamify, integration_complexity, mac_mps_support_verified, memory_estimate_measured_or_unknown, offline_risk, privacy_risk, scientific_risk, testable_hypothesis, baseline_control, evaluation_dataset, benchmark_protocol, effort_engineering_h, effort_data_h, time_machine_h_measured_or_unknown, impact_estimate_exploratory, evidence_level, verdict, reason`.

**ציון evidence:**
- E0 — התגלה בחיפוש בלבד; אין אימות.
- E1 — קיום repo ו־README/קוד/רישיון נקראו.
- E2 — התקנה ובדיקות smoke מקומיות, הגדרות תלויות/משקלים הוכחו.
- E3 — השוואת benchmark מבוקרת מול Taamify על אותם IDs, seeds, API/feature conditions.
- E4 — שיפור משוחזר על קוראים אמיתיים חדשים עם מבחן נעול מראש; אין להעניק E4 ללא אמת חדשה.

דירוג A = מועמד לפיילוט מיידי ללא חסם מחקרי; B = אחרי אישור רישיון/נתונים; C = מידע למחקר/מאמר; D = לדחות. לדווח גם סיבת דחייה, לא למחוק תוצאות שליליות.

**Weight and dataset restrictions are separate from code license.** GPL/AGPL, CC-BY-SA, CC-BY-NC, research-only weights and dataset audio conditions require separate review. אין להסיק "מותר מסחרי" מ־MIT של קוד אם weights או datasets שונים. אין להעתיק קוד לפרויקט או לשנות license לפני הכרעה.

## 5. שיטת מחקר ובדיקת טענות

**Pass 1 — Evidence inventory:** coverage מכוונת של 100+ candidates אם קיימים; dedupe forks, archive flags, dated commit and repo license. בצע חיפוש לפי topic, code, papers with code, citations, GitHub issues.  
**Pass 2 — Direct relevance:** קרא README, source files, installation, current issues, datasets/model card, evaluate if identical function exists in Taamify already; discard duplicated stack.  
**Pass 3 — Top 20:** detailed technical review, native Apple compatibility and license decision.  
**Pass 4 — Top 5–10:** isolated feasibility in sandbox/venv, no production checkpoint switches; generate pass/fail and resource profiling **only after allowed**.  
**Pass 5 — Top 3–5:** plan controlled experiments with null hypothesis, ablation, same-split, per-speaker CI, ≥3 seeds for finalists, label and consent checks. Do not open sealed holdout repeatedly.  
**Pass 6 — Final architecture:** choose minimal set of modules with measurable advantage; **do not** glue 100 OSS packages into product.

פרסם *negative results*; benchmark by validation/calibration before final test; synthetic derivatives stay in same split/source-root. Pre-register number of comparisons and primary outcome to avoid winner's curse. Every integration cites actual repo commit/tag, package version and license decision. Estimate throughput only after measured minutes/epoch on target machine.

## 6. תוצר סופי קשיח: לא להסתפק ב"אספתי מידע"

Create under `docs/research-os/github-study/`:
- `FINAL_RESEARCH_REPORT.md` — abstract, existing system, full landscape and coverage, direct prior art comparison, evidence-based synthesis, ranked 20, top 5 experiments, rejected options, legal/data constraints, academic novelty, product roadmap, unresolved questions, citations with URLs and access dates.
- `REPO_EVALUATION_MATRIX.csv` (or JSON) — every candidate E0–E4 with explicit unknowns and primary source.
- `ARCHITECTURE_DECISION_RECORDS.md` — ADR per adopted/rejected technology (Alternatives / Decision / Tradeoff / Reversal).
- `PILOT_PROTOCOLS.md` — exact reproducible tests (baseline, criteria, compute, stop-loss, expected failure modes, rerun commands) for each top prospect.
- `ACADEMIC_RESEARCH_SYNTHESIS.md` — relation to prior art, claim boundaries, contribution, paper sections, citations to original papers.
- `PRODUCT_IMPACT_MAP.md` — which measured problem is solved (A1/A3/A4, new reader, latency, calibration, real errors); no made-up metric gains.
- `IMPLEMENTATION_BACKLOG.md` — issue-sized ordered tasks with dependencies, effort ranges, owners, acceptance criteria and gating.
- `LICENSING_AND_DATA_REGISTER.md` — code / model / audio / texts rights distinguished; unknowns block commercial use.
- `EXECUTIVE_DECISION.md` — 1–2 pages with 3 recommended technical routes, 1 recommended choice, evidence and explicit disconfirmation conditions.
- Machine-readable `research_os/config/github_ecosystem_review.json` with validation script and no personal data.

**Every file** must distinguish `verified`, `reported_by_repo`, `hypothesis`, `not_checked`. Include date of evidence and code SHA where relevant. Never claim integration without a runnable test result.

## 7. סדר התחלה וסדרי עדיפויות

1. S0: current checkout, code/data/served model audit and rights for Ben13. Do not replace production or leak the corpus.
2. First examine specialized Torah repositories **staviv/cantillation, kylemath/cantillate, Moriahise/taamimflow_project**; they may materially change prior-art claims. A project finding a textual taam is *not* proof of performed melodic correctness.
3. First technical cluster for feasible test: `maxrmorrison/torchcrepe` or `interactiveaudiolab/penn` and melody-contour evaluation, `MontrealCorpusTools/Montreal-Forced-Aligner`/`m-bain/whisperX` with Biblical Hebrew feasibility, `s3prl/s3prl` / frozen WavLM, `iver56/audiomentations` robustness. Always test against existing baseline first.
4. Study VC as controlled secondary track, **after** consent and human melodic fidelity audit. Archived Seed-VC is not immediate deploy dependency.
5. Gather new real readers and authentic mistakes in parallel with reading source; do not block on code research.
6. Never present 51 theoretical fits / 1,350 fit-epochs as measured machine hours.
7. Send owner a short decision package before any expensive jobs, third-party data transfer, model or threshold changes.

## 8. Acceptance criteria

- At least 100 actual relevant candidate repositories **or** documented reason fewer meet the search criteria; count forks only if material differences.
- Every examined candidate has working exact URL, date/archived/license status with links or `unknown`, planned leverage, blocker and verdict.
- >=20 detailed repository reviews where relevant; >=5 reproducible pilot proposals, actual pilots only when local tools/data/authorization are present.
- Top 3 findings explain why not rely solely on the current one-reader Stage A accuracy and how to measure benefit on independent new readers.
- Separate scientific novelty/prior-art/assumptions from product benefits; quantified error/coverage and **genuine** learner-error results only when collected.
- Final research report is written as a self-contained, reviewer-quality document with explicit limitations; no placeholders disguised as facts.
- Never automatically merge/deploy/push raw data. Open a draft PR with report and evidence for review.

**Use longlist as starting hypotheses only; the finished report is complete only when the evidence requirements above are met.**
