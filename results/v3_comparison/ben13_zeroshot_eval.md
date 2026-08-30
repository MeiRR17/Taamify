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