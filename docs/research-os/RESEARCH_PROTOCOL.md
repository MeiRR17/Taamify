# Taamify: preregistration proposal / פרוטוקול מדעי v0.1

**Status: Draft; no new empirical result.** This is an academic study protocol for review, *not* a claim that experiments were performed. Study scope = Jerusalem-Sephardi Torah cantillation only, with distinguishable product and scientific endpoints.

## Scientific research problem

Given an acoustic recording of a Torah word or phrase, can a learned representation distinguish the *performed* cantillation from identity of reader, lexical content, room, microphone and expectations from the written verse? Can pretrained speech/music features, explicit F0 contours or synthetic speaker augmentations improve **generalization to genuinely new readers**, beyond the original pitch-shift augmentation?

### Research questions and falsifiable hypotheses

| ID | Question | Null hypothesis | Primary comparison |
|---|---|---|---|
| H0 | Does the current model replicate its documented result through the real API? | Its claimed metrics fail under independent replay or matched split | Stage A replay vs frozen documented summary |
| H1 | Do explicit F0, prosody and/or multiresolution windows improve taam identification? | No better balanced accuracy than matched DeepCNN | P01/P02/P06 vs P00 |
| H2 | Do frozen pretrained speech embeddings improve unseen-speaker generalization? | WavLM/W2v-BERT not better on paired readers | P03/P04/P11 vs P00 |
| H3 | Do music encoders add value for melodic accents? | MuQ has no incremental benefit after matched control | P05 vs P01/P00; noncommercial research only |
| H4 | Is voice conversion better than pitch shift + existing augmentation? | No benefit over exact pitch-shift-only control | P08 vs same base/pitch control |
| H5 | Is teacher-referenced verification more reliable than class-only verification? | Same false-accusation rate and true-error detection | P07 vs current VerifyHead on real wrong/correct examples |
| H6 | Can speaker-invariant representations improve cross-reader transfer? | Domain-adversarial/contrastive models do not improve speaker-held-out error | P09/P10 vs matched frozen base |
| H7 | Can distilled student serve locally without meaningful error increase? | Student misses pre-set accuracy/calibration/latency requirements | P12 vs validated teacher |

**Scope guard:** already demonstrated in project notes — pitch shift increases high-pitch robustness, 7-model ensemble underperformed best model, Torah vs Haftara mixtures produced asymmetric results, old random-split/incorrect-file coupling created misleading estimates. These are priors to control against, not novel results to rediscover.

## Data stewardship and sampling

- Build a gold evaluation set with independently labeled, **naturally correct and incorrect** spoken performances, in the same musical tradition; collect multiple real readers and repeated passages.
- An eight-speaker target (6 development + 2 final holdout) is only a planning starting point, NOT a claim of sufficient statistical power. Power/sample-size decision should be made after a pilot estimating between-reader variance; increase speakers if uncertainty remains wide.
- A `reader_id` is pseudonymous, but a voice remains potentially identifying data. Consent, permitted reuse, deletion, minors/guardian consent, and audio source rights must be documented **before** data joins a train job.
- Annotate `taam_intended`, `taam_performed`, correct/incorrect/ambiguous, `verse_id`, `word_id`, `root_recording_id`, `speaker_id`, `alignment_confidence`, `tradition`, `consent_scope`, `recording_device`, provenance and QC status.
- At least a subset independently annotated by two competent readers; report agreement and adjudication. Do not use model predictions as "ground truth".
- No mixing Sephardi-Jerusalem with other cantillation traditions as extra examples within this study.

## Independent units and splits

- **Development:** speaker-grouped cross-validation (Leave-One-Reader-Out), and passage/recording grouping inside folds. The source recording and all derived/synthetic audio must reside in the same arm.
- **Model selection/thresholds:** development folds only. Never tune to hidden test. If there are too few speakers, defer confirmatory claims.
- **Final holdout:** previously unseen real speakers/recordings, untouched until hypotheses, architectures, settings and thresholds are frozen in a Git commit.
- **Voice conversion:** only train-allowed, consented target speaker references; preserve original source root, never use final-test speaker exemplars in training **even as voice conversion references**.
- **Known text:** keep acoustic-only and context-aided tracks separate; compare actual performed accents to expected canonical text without letting canonical text substitute for listening.

## Primary / secondary outcomes

**Primary scientific outcome:** per-speaker balanced accuracy on new readers, reported as *paired difference* vs P00 / relevant exact augmentation control on identical test units. Include macro-F1, per-taam recall and taxonomy-aware confusion analysis as secondary. Report detection coverage and abstention fractions.

**Primary product outcome:** sensitivity to human-verified actual errors while respecting a <=2% false-accusation target selected on development/calibration only; confidence calibration, unclear-audio rejection, end-to-end delay and score consistency also mandatory.

**F0-specific audit:** evaluate voiced/unvoiced boundaries, F0/octave-error rate, relative semitone contour, temporal landmarks, phrase duration, stress location, phoneme preservation and error localization. Pitch-shift preserving the melody is a *control*, not equivalent to an independent singer. For VC, reject synthetic clips that introduce taam-altering contours or unnatural lexical timings.

## Statistics and preregistration

1. Freeze question priority, exclusion rules, metrics, grouping, stop/go decisions, model seeds, teacher references and split hashes *before* looking at sealed test results.
2. Report uncertainty by **speaker cluster**, not by treating every word from one reader as an independent replication. With few speakers, uncertainty may be large; present descriptive intervals and limits honestly.
3. For multiple experimental arms, designate H2 and H4 as proposed primary confirmatory claims, or explicitly mark all arms exploratory until protocol review. Use an appropriate multiplicity procedure for any confirmatory family; report both raw and adjusted values.
4. Distinguish algorithm variance across seeds from variation across human readers. At least 3 model seeds for finalists; do not promote tiny single-seed gains (prior project estimated ~1.17pp seed SD).
5. Negative controls: random/shuffled labels, same-width random windows, matched feature size, zero/noise pitch, same-word/different-speaker, same-speaker/different-taam, reference audio quality, pitch-shift-only, synthetic artifacts detector, speaker probe on taam embedding, and acoustic-only vs context-aware.
6. A **practical importance margin** should be preregistered (e.g., +2 percentage points balanced accuracy as an initial target, to be reassessed based on reader variance) independently of statistical significance. The default is NOT a promise of achievable gain.
7. Model comparison must report *failed* experiments and costs, not just winner curves.

## Sequential experiment stages and exit criteria

| Phase | Procedure | Exit evidence |
|---|---|---|
| S0 Audit | Reproduce Stage A through live API with exact code+data+checkpoints; verify file joins and splits | Immutable audit record; no unexplained mismatch |
| S1 Feasibility | Per-pipeline forward pass, 1 pilot seed, M1 Max hardware throughput, peak unified memory and source integrity | Real measured minutes/epoch; candidates without OOM and valid quality |
| S2 Development | Cross-reader controlled study after collecting genuine new voices; baseline + finalists, 3 seeds × predefined folds | Paired reader-level metrics, error analysis, calibration |
| S3 Final | Freeze code, model, thresholds; single opening of new real-speaker holdout | Final estimates and uncertainty without repeated selection |
| S4 Knowledge | Interpret acoustic vs lexical/context and academic domain implications | Manuscript/technical report, reproduction instructions |
| S5 Product | Independently validate real user recordings, live endpoint, no false-feedback drift | User-reviewable product gate; not automatic merge/deploy |

## Public/academic deliverables

- Methods: versioned preregistered protocol; data dictionary; anonymized aggregate manifest; legal/consent statement; model datasheet and limitations.
- Results: per-speaker/taam tabulation; ROC/PR + calibration; ablations; variance; confusion structure; falsification/negative-control outcomes; test-set accounting.
- Engineering: reproducibility lockfile; code commit SHA; checkpoint provenance and hashes; exact train/validation/test IDs maintained locally; device benchmark; sanitized logs.
- Scholarly writing: paper outline, related work on cantillation/speaker representations, statistical appendix, product limitations.
- Real audio and confidential identifying material stay in local secure storage, not GitHub or shared Drive.

## Starting references

- WavLM (Chen et al., 2022): https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/
- Meta W2v-BERT 2.0: https://huggingface.co/facebook/w2v-bert-2.0
- Seed-VC, original GitHub archived Nov 2025: https://github.com/Plachtaa/seed-vc

**Approval needed before execution:** define real eligible speakers and independent annotators, freeze preregistration, verify production checkout, profile actual machine throughput. This technical PR alone does not do those actions.
