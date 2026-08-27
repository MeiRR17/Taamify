# Milestone 3, Stage A — verify against known reference, not blind classification

The real product flow: the user picks a parasha first, so at inference time
the expected ta'am for every word is already known from Sefaria's tagged
text. The task isn't blind 11-way classification, it's verification — "does
this audio match the expected mark?" This is measured as its own number,
never folded into the raw 11-way accuracy the earlier milestones report.

## Method

`15_binary_verify_model.py` reuses `14_sequence_model.py`'s
`embed_everything()` (frozen per-word CNN → embedding), then trains a small
MLP head on `[embedding ; one-hot(candidate class)] → match logit`. The
corpus only contains correct readings (no labelled mistakes), so training
pairs are constructed directly: every real word gives one positive pair (its
own true class) and one negative pair (a different, randomly sampled wrong
class) — a balanced verification task. This still trains the right function
(does embedding *e* look like class *c*, for any *e*/*c* pair), which is what
deployment needs regardless of how the training pairs were built — at
inference the query is always the text's actual expected class, correct most
of the time and occasionally wrong when a learner errs.

Same LORO discipline as every other milestone: the CNN checkpoint used for
each fold was itself trained without that fold's held-out reader, and the
verify head's own train/val split is by verse, matching
`14_sequence_model.py`.

## Result

| Fold | Raw 11-way top-1 | Stage A binary accuracy | Δ | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| R1 held out | 46.9% | **77.8%** | **+30.9** | 0.847 | 0.678 | 0.753 | 0.854 |
| R2 held out | 82.7% | **94.4%** | **+11.7** | 0.950 | 0.936 | 0.943 | 0.985 |
| R8 held out | 78.3% | **93.0%** | **+14.7** | 0.932 | 0.929 | 0.930 | 0.980 |
| **Mean** | **69.3%** | **88.4%** | **+19.1** | | | | |

**The hypothesis holds, clearly and on every fold** — "matches expected" is a
much easier discrimination than "which of 11," most dramatically on R1, the
fold every earlier milestone found hardest (Milestone 1: R1 stays flat
regardless of what's in the training pool). Verification framing recovers
30.9 points there specifically — the product-shaped task is not just
theoretically easier, it's easiest exactly where blind classification
struggles most.

**Honest caveat on R1's balance:** precision (0.847) meaningfully exceeds
recall (0.678) on the hardest fold — the model is more confident when it
says "match" than when it says "no match," i.e. it under-calls matches more
than it under-calls mismatches on this fold. R2/R8 are well balanced
(precision ≈ recall). Worth watching if Stage A is tuned further: a
product that's supposed to tell a learner "you got it right" should not be
systematically stingy with that verdict on exactly the reader profile it
generalizes worst to.

**Test-set balance note:** the binary test set here is constructed 50/50
match/no-match, which measures raw discriminative power fairly but does not
reflect real deployment prevalence (most readings from a competent reader
are correct, so real "no-match" queries would be rarer than 50%). Accuracy
numbers above should not be read as "the app will be right 88% of the time
in practice" — that depends on the true error rate of whoever is reading,
which this corpus (all proficient readers) cannot measure.

## What's next (Stage B)

Stage B — classify *which* of the 11 marks was actually sung, only on the
subset Stage A flags as no-match — reuses the existing per-word CNN
unchanged. Not yet run as its own measured number on this corpus, since the
corpus has no real no-match examples to test it on (same limitation noted
above). For now, Stage B's expected accuracy is the raw 11-way LORO number
already reported in Milestone 0/1, applied only to the harder subset.
