# Ben13 zero-shot evaluation — Stage A on a voice never in training

The production Stage A verify-head (Milestone 5) trained and measured on R1/R2/R8. Ben13's segmented clips (`Data_Prep/segment_ben13_clips.py`, Milestone 1C) are a genuinely independent test: a fourth voice, studio-clean, that appears nowhere in training.

**Scope**: 97/114 segments have a true class inside the production model's top-11 vocabulary; the other 17 (Darga, Geresh, Qarney_Para, Segol, Yerach_Ben_Yomo, Zarqa) have no slot in the verify-head's one-hot space and are excluded, not silently dropped. Siluk never appears (verse-final, doesn't occur mid-phrase in a combo clip). Same pairing convention as Milestone 3: every segment gives one positive pair (true class) and one random-wrong-class negative.

| Set | Pairs | Accuracy | Precision | Recall | F1 | Match conf. (mean/median) | No-match conf. (mean/median) |
|---|---|---|---|---|---|---|---|
| All 97 segments (194 pairs) | 194 | 0.686 | 0.743 | 0.567 | 0.643 | 0.523 / 0.590 | 0.248 / 0.130 |
| High-confidence only (76 segments, 152 pairs) | 152 | 0.678 | 0.755 | 0.526 | 0.620 | 0.494 / 0.546 | 0.237 / 0.127 |

For reference, Milestone 3's R1/R2/R8 LORO mean was **88.4%** (range 77.8-94.4%).

## This doesn't hold up, and it's worth being precise about why

**Accuracy drops ~20 points (88.4% → 68.6%), driven almost entirely by
recall** — 52.6-56.7%, meaning the model fails to confirm roughly *half* of
genuinely correct matches on this audio. Precision holds up reasonably well
(74-76%), so when the model does say "match," it's usually right; it's
specifically under-confident on true matches. The high-confidence subset
(discarding the 21 lowest-CTC-score segments) barely moves the number
(68.6% → 67.8%) — so this isn't primarily a segmentation-quality artifact.

**But this is not a clean "unseen voice" test, and shouldn't be read as one.**
Milestone 1C already established what these clips actually are: the teacher
*singing the ta'am's own name* (in the Sephardic naming convention), not
chanting continuous Torah verse text. That's a genuine confound layered on
top of the unseen-voice question — this test simultaneously changes the
voice *and* the content genre (isolated single/few-word pedagogical
utterances vs. continuous chanted prose), where Milestone 3's LORO folds
only ever changed the voice. A drop here is expected on general
distribution-shift grounds and doesn't contradict Milestone 3's result that
Stage A generalizes well across three *natural-reading* voices — it measures
a different, narrower question: does it also hold up on a different kind of
recording entirely. It doesn't, at least not to the same degree.

**What this is actually useful for**: it's a real, now-measured boundary of
where Stage A's 88.4% applies (continuous natural reading, three voices) and
where it hasn't been shown to hold (isolated didactic single-word audio).
Treat 68.6% as the honest number for *that* narrower claim, not as a
contradiction of Milestone 3's headline result.

## Follow-up: does training on BEN13AUG actually help? (single-fold pilot)

Prompted by the same question this file answers — is Ben13 worth more than
reference material — ran a direct pilot: R1 held out, 25 classes, `deep_cnn`
+ weighted CE, with vs. without `BEN13AUG` (the 60 synthetic examples for
Shalshelet/Yerach_Ben_Yomo/Mercha_Kefula/Qarney_Para prepped in Milestone 1C)
folded into training.

| Condition | Classes | Accuracy | Balanced accuracy |
|---|---|---|---|
| Without BEN13AUG | 23 (R1/R2/R8 alone have zero real Shalshelet/Qarney_Para instances) | 41.0% | 34.8% |
| With BEN13AUG | 25 | 38.6% | 32.4% |

**It made things worse, not better — and the benefit it was meant to buy is
unmeasurable on this fold anyway.** R1's test set has 0 real instances of
Yerach_Ben_Yomo/Shalshelet/Qarney_Para and exactly 1 of Mercha_Kefula, so
there's nothing to confirm the model actually learned these classes against.
Meanwhile recall declined on 13 of the 21 shared classes (improved on 8) —
a broad, if not universal, regression on the classes that *can* be measured.
Plausible mechanism: 15 near-identical synthetic variants (only pitch/time
varied) of a single real clip is low-diversity signal that adds confusable
noise to the shared softmax without giving the model a real new decision
boundary to learn.

**Verdict: do not fold BEN13AUG into training as currently built.** This is
a single-fold pilot (R1 only), not a full LORO validation, but the effect is
broad enough and the potential upside unmeasurable enough that the honest
call is to hold, not to extrapolate a full 3-fold sweep to confirm a result
that's already this one-sided. If the rare classes are revisited, the more
promising paths are real additional recordings, or richer augmentation
(noise/reverb/EQ variation, not just pitch-shift/time-stretch of one clip) —
not more synthetic copies of the same single source.

One genuinely useful thing this pilot caught along the way: `10_train_pipeline.py`'s
`classification_report`/`confusion_matrix` calls crashed the first time this
ran, losing an entire training run's checkpoint and metrics, because a class
with real training volume can still have zero instances in one particular
LORO test fold — a real edge case only surfaced once classes are pushed into
the long tail. Fixed (`labels=range(len(class_names))` pinned explicitly on
both calls) and confirmed the rerun completes cleanly.