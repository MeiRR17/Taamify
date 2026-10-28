# Taamify — אבחון מקדים של אקו־סיסטם GitHub והחלטות מחקר
**28.10.2026 | Preliminary, evidence-linked. Not final multi-repository audit.**

## Executive Summary

כדי להגיע למוצר אמין ולמאמר מדעי, Taamify **לא** זקוק להעתקת 100 ספריות. הוא זקוק ליכולת להבחין בין טעם שבוצע בפועל לבין מה שהיה צפוי בטקסט, לעבור מקורא יחיד לקוראים חדשים, ולהציג משוב אמין גם על טעויות אמת. רשימת ה־116 היא **unreviewed search pool**. להלן מסקנות ראשוניות מפרויקטים שנקראו באופן ישיר לצד מצב הפרויקט המקומי. E1 = README/license evidence only; no working integration executed.

## 1. Prior art שאסור לפספס

### P0.1 — [staviv/cantillation](https://github.com/staviv/cantillation)

ה־README מציג מערכת מבוססת Whisper להפקת תמלול תורה יחד עם טעמי המקרא, מדדי cantillation וממשק התנסות. תוצר זה **קרוב מאוד לבעיה שעליה Taamify רוצה לפרסם**: זיהוי טעמים, תיקונים ולימוד עצמי. הריפוזיטורי מציין Ben13 ו־PocketTorah כמקורות. אין להסיק שמודל ASR עם תחזית טעמים מתוך אות קול מאמת ש"טעם שבוצע בפועל" אינו פשוט משוחזר מתוך הטקסט המוכר; צריך control מסוג audio-only vs text-only vs swapped audio/verse. השאלה על novelty חייבת להיכתב ביחס לעבודה הזו ולא כטענת "ראשונים בעולם".

- בחן קוד `cantilLocations_evaluation.py` / normalization, הקלטות, split, label definition, train-inference paths.
- הורד רק קוד ציבורי מותר, לא קורפוס שקיומו על GitHub אינו מוכיח זכויות מסחריות.
- מטה־נתונים מקוריים והצגת תוצאות [Technion SIPL 2025](https://sipl.ece.technion.ac.il/projects/project-details/?prj_id=11111) מדווחים הרחבת קורפוס ל־337 שעות ותוצאות על נוסחים שונים; אלו **טענות שלהם**, לא שכפול עצמאי. אסור להשוות F1 שלהם ל־A1 85.7% של Taamify כאילו המדדים זהים.
- תוצר מחייב: direct prior-art matrix, מה Taamify תורם מעבר לתמלול קאנוני, ו־text-leakage ablation.

### P0.2 — [kylemath/cantillate](https://github.com/kylemath/cantillate)

ה־README מפרט **guided mode** עם חמש רמות תרגול, ניהול פרשה לפי תאריך, טקסט, הקלטות, waveform/pitch shapes, מקטעי מילים וקליפים. זה חומר חשוב ל־**product benchmark**, ויכול לשנות את תכנון הממשק של Taamify באופן מהיר יותר מעוד CNN.

- חקור את פעולות microphone, pitch shape, feedback, cut-and-repeat, פרשות, טקסט מתוך ספר תורה, תזמון כללי.
- חלק מהקלטות מגיעות מ־PocketTorah לפי README, ברישיון שמחייב בדיקת שיתוף־זהה וייחוס; אין להעתיק assets ללא בדיקה.
- בנה טבלת feature parity (Taamify today / Cantillate / proposed pilot) והערך זמן יישום של הפיצ'ר הקטן ביותר שמגדיל תועלת לומד.

### P0.3 — [Moriahise/taamimflow_project](https://github.com/Moriahise/taamimflow_project)

תיעוד הפרויקט מציין stubs וחלקי audio synthesis שטרם הושלמו. הרכיבים הקיימים מעניינים כעיצוב מסלול תרגול, תצוגת תיקון, וחיבורי Sefaria; אינם הוכחה למנוע אימות טעויות. השווה architecture/UI בלבד וסקור רישיון הקוד והנכסים.

## 2. חמש המלצות לפיילוט ממוקד (תכנון בלבד)

| Rank | Hypothesis / leverage | Project(s) | Controlled test | Go/No-Go |
|---:|---|---|---|---|
| 1 | contour-aware verification improves rejection of *real* taam errors | [torchcrepe](https://github.com/maxrmorrison/torchcrepe) / [PENN](https://github.com/interactiveaudiolab/penn) + [librosa](https://github.com/librosa/librosa) | paired source/teacher F0 tracking; no-text-only and same-word different-taam controls; compare current VerifyHead at matched false alarm and coverage | **No claim** before genuine wrong readings and readers. Gate F0 quality on sung Hebrew |
| 2 | alignment reduces actual end-to-end failures | [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) / [WhisperX](https://github.com/m-bain/whisperX) | hand labeled word/syllable boundaries for 8-parasha sample; mean/tail boundary errors and API accuracy with same checkpoint | reject if Biblical Hebrew lexicon/inference unsupported or improvement falls within noise |
| 3 | frozen speech embeddings generalize beyond Ben13 | [S3PRL](https://github.com/s3prl/s3prl) / [SpeechBrain](https://github.com/speechbrain/speechbrain) / [Transformers](https://github.com/huggingface/transformers) | WavLM frozen probe vs DeepCNN on speaker-disjoint real readers; identical splits, 3 seeds, peak M1 Max RAM | defer confirmatory until consented reader ≥2; first only resource feasibility |
| 4 | additive realistic augmentation helps real-device robustness | [Audiomentations](https://github.com/iver56/audiomentations) / [torch-audiomentations](https://github.com/iver56/torch-audiomentations) | same source and target real phones, RIR+codec vs pitch-shift-only; human taam preservation, speaker-root grouping | stop on corrupted labels, false accusation drift, or worse out-of-reader result |
| 5 | user experience can improve before expensive training | [Cantillate](https://github.com/kylemath/cantillate) / [WaveSurfer.js](https://github.com/katspaugh/wavesurfer.js) | 5 students / teacher review pilot if consent/ethics; compare practice completion, comprehensibility and feedback delay | reject if harms accuracy, privacy or time-to-feedback |

These are **test proposals**, not successful Taamify experiments. Current Stage A follows Ben13 only. Count true reader identities instead of treating augmentation variants as independent speakers.

## 3. Risk register

| Risk | Concrete evidence or trigger | Decision now |
|---|---|---|
| Prior art conflict | staviv/cantillation + 2025 Technion project; own feature overlap with Cantillate | write precise novelty table before drafting article claim |
| Label contamination | memorizing verse-to-taam sequence instead of analyzing acoustic performance | acoustic-only model and swapped-verse negative control mandatory |
| Speaker overfitting | Stage A single reader; two external real examples previously failed | collect real same-tradition voices and manually marked wrong readings |
| True false warnings | stage replay A3=2.1% > threshold 2%; grammar combined false flag 13.7% candidate 2.8% | insist on end-to-end API calibration with genuine wrong-reading sensitivity |
| Synthetic identity leakage | many VC clones from one reader remain same underlying source | source-root and voice-reference split guards; consent and expert F0 audit |
| Copyright / licensing | [AudioCraft](https://github.com/facebookresearch/audiocraft) MIT code, CC-BY-NC weights; Seed-VC GPL code and archived; RVC code MIT but third-party weights distinct | no commercial use until per-layer rights register completed |
| Deprecated API | [PyTorch TorchAudio forced alignment notice](https://github.com/pytorch/audio/blob/main/examples/tutorials/ctc_forced_alignment_api_tutorial.py) reports removal in v2.9 | do not adopt stale tutorials; lock package versions |
| Compute waste | M1 Max unified memory shared with Ollama; actual minutes/epoch not yet measured for new families | profiling before broad sweep, no second queue manager |
| Scientific claim drift | old random-split and proxy mistakes are not independent evidence | preregister controls, identify evidence level E0–E4, retain negative results |

## 4. Proposed limited-scope three-month path (not a promise)

**Weeks 1–2:** evidence/repository audit, license matrix, read existing code, prior-art comparison, gold dataset rights check, hand-labeling protocol, collect speaker commitments. Do not spend 2 weeks installing 100 repos.  
**Weeks 3–4:** reproduce Stage A API; manually label genuine correct/wrong recordings; benchmark F0 & alignment feasibility; continue channel calibration with unchanged live thresholds.  
**Weeks 5–8:** only candidates that pass trials; controlled matched experiments on real new speakers when available; reader-grouped CIs; product guided practice baseline with teacher validation.  
**Weeks 9–12:** acceptance reviews, latency and accessibility, rollback, candidate model if proven; preregistered research results and draft article, release beta *only if* minimum evidence/rights/UX gates clear.

The calendar is conditional on real voice acquisition, consent, successful Hebrew alignment, model compatibility and human teacher review. It is possible that a credible beta is not justified within 3 months; accuracy and safety take priority over schedule.

## 5. Early decisions

1. **Do not replace the existing Taamify service.** Any other toolkit starts as an isolated adapter and PR.
2. **Treat direct prior art as priority above fashionable foundation models.** The user seeks a paper with actual contribution, not just "we trained another neural net".
3. **Review 116 candidate leads, but integrate probably only a few.** The final report must rank them by direct measurable leverage; do not fill an adoption quota.
4. **No full scientific novelty verdict yet:** prior-art literature and repo code analysis remain pending, and no independent tests of third-party models were performed in this preliminary assessment.
5. **No audio/data upload** to external platforms or GitHub; local corpus restrictions and privacy of minors control all plans.

## Evidence trail (primary documents opened/read this session)

- [Taamify Stage A](https://github.com/MeiRR17/Taamify/blob/v3-tradition-finding/PLAN_STAGE_A.md)
- [Taamify Task 1 calibration](https://github.com/MeiRR17/Taamify/blob/v3-tradition-finding/docs/research-os/TASK1_CHANNEL_CALIBRATION.md)
- [staviv/cantillation README](https://github.com/staviv/cantillation)
- [Moriahise/taamimflow_project README](https://github.com/Moriahise/taamimflow_project)
- [kylemath/cantillate README](https://github.com/kylemath/cantillate)
- [Seed-VC README](https://github.com/Plachtaa/seed-vc)
- [RVC README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Audiomentations README](https://github.com/iver56/audiomentations)
- [PENN README](https://github.com/interactiveaudiolab/penn)
- [AudioCraft README](https://github.com/facebookresearch/audiocraft)
- [Technion SIPL project 2025](https://sipl.ece.technion.ac.il/projects/project-details/?prj_id=11111)
- [Quantified Cantillation](https://quantifiedcantillation.nl/)
