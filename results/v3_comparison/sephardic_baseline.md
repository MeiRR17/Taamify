# Sephardic-Jerusalem baseline — current corpus, current code

Everything below is fresh: retrained on the post-rebuild corpus
(`aligned_dataset_merged.json` as of this round — phonology fixes, verse-level
realignment), R1/R2 only. Numbers from before this round (including the old
sequence-model report) are stale and should not be cited.

## Ceiling: random split (same reader in train and test)

| Model | Accuracy | Balanced Acc | MCC | AUC |
|---|---|---|---|---|
| resnet18_tl | 75.3% | 77.5% | 0.722 | 0.965 |

This is the "can the model learn ta'amim at all" number. It is not the
product metric.

## The real metric: unseen-reader generalization (LORO)

| Model | R1 held out | R2 held out | Mean ± Std |
|---|---|---|---|
| deep_cnn | 46.9% | 55.6% | **51.2% ± 4.4** |
| resnet18_tl | 45.8% | 47.0% | 46.4% ± 0.6 |

**deep_cnn generalizes better than resnet18_tl** despite resnet18_tl winning
the random-split ceiling by 24 points. The stronger ImageNet-pretrained
backbone overfits to reader-specific acoustic detail; the smaller model
doesn't have the capacity to. deep_cnn is the base architecture for
everything below.

## Sequence context: does verse grammar help the unseen reader?

Per-word CNN (deep_cnn) vs a BiLSTM tagger over frozen CNN embeddings
(verse-level context) vs the same BiLSTM with a linear-chain CRF head, on
the identical LORO test words in each fold.

| Fold | CNN alone | + BiLSTM | + BiLSTM + CRF |
|---|---|---|---|
| R1 held out | 46.8% | 57.4% (+10.6) | 58.0% (+11.2) |
| R2 held out | 55.5% | 70.5% (+15.0) | 70.7% (+15.3) |
| **Mean** | **51.2%** | **64.0% (+12.8)** | **64.4% (+13.2)** |

Verse context alone recovers most of the way from "unseen reader" toward the
75% same-reader ceiling. The CRF adds a further +0.4 points on average — real
and consistent in direction across both folds, but small; the BiLSTM already
captures most of the sequential structure through its own recurrence, and the
CRF's explicit hard transition constraints add comparatively little on top.
Not yet multi-seeded — treat the CRF's marginal gain as suggestive, not
proven, until it is.

CRF motivation (measured on this corpus, not assumed): of 5,785 verses, 5,581
carry exactly one silluq and 5,424 exactly one etnachta, and *none* carries
two. The grammar a CRF has to learn is close to deterministic.

## A caught bug, worth recording

The sequence-model script's embedding cache originally keyed on the
checkpoint's filename stem. Two different LORO folds both save to a file
named `deep_cnn_best.pth` in different directories — the R2-holdout run's
first attempt silently loaded the R1-holdout run's cached embeddings and
reported 84.8% "CNN alone" accuracy on R2, evaluating the wrong model (one
that had R2 as training data) rather than erroring. Caught because it
contradicted the already-measured 55.6% LORO number by 29 points — cross-
checking a new number against an existing one is what surfaced it. Fixed:
the cache key now hashes the checkpoint's resolved path plus the reader set.

## Milestone 1: a third reader (R8) — coverage, not reader count, is the lever

Added R8 (Nissim Shaked), same tradition, 33/54 parshiyot, passing Gate 2
with a better keep-rate and CTC score than both R1 and R2. Full 3-reader LORO
(R1, R2, R8), same corpus, two independent architectures:

| Fold | deep_cnn | resnet18_tl |
|---|---|---|
| R1 held out | 46.9% | 47.1% |
| R2 held out | **82.8%** | **81.4%** |
| R8 held out | **78.5%** | **77.5%** |
| Mean ± std | 69.4% ± 16.0 | 68.7% ± 15.3 |

R2 and R8 held out jumped 27-33 points over the 2-reader baseline (51.2%
mean); R1 held out barely moved at all. That asymmetry was suspicious enough
to investigate before trusting it — an unseen reader scoring *above* the
same-corpus random-split ceiling (77.7%, deep_cnn) is a leakage red flag on
its face. Ruled out, in order: shared audio file paths between readers
(none), duplicated feature arrays (none — R2_X.npy and R8_X.npy are distinct
data), a bug in the LORO split code (`make_loro_splits` is a strict boolean
mask by reader identity), metric misattribution in `11_run_loro.py` (each
fold trains and evaluates from its own local copy), acoustic similarity
between R2/R8 that isn't shared with R1 (pitch-contour distance is uniform
across all three pairs, near the within-reader noise floor), and run-to-run
variance (retrained the R2-holdout fold from scratch — reproduced to 4
decimal places). Then cross-checked with a second, architecturally unrelated
model (resnet18_tl, ImageNet-pretrained transfer learning vs. deep_cnn's
from-scratch 5-block CNN) — the same pattern reproduced almost exactly,
which rules out a model-specific artifact.

The actual explanation: **R1 covers 49/54 parshiyot (91% of the Torah); R2
covers 30/54 and R8 covers 33/54.** A training pool without R1 (R2+R8 only)
has much narrower word/context coverage even combined, so it generalizes
poorly to *any* unseen reader — including R1's own, full-coverage test set.
A training pool that includes R1 gets near-complete Torah coverage and
generalizes well to whichever single reader is held out. This means the
mean-of-three-folds framing is somewhat misleading on its own: it isn't
"three readers reliably beats two," it's "training coverage of the text is
the real lever, and reader count is a proxy for it that happens to work here
because R1 is unusually complete." The practical implication for Milestone 1
going forward: prioritize finding readers with broad parasha coverage (or
combining several partial readers to jointly cover the Torah) over simply
adding more readers with narrow, overlapping coverage.

R1 held out staying flat at ~47% regardless of what's in the training pool
is itself worth taking seriously: it suggests R1's test set contains
word/context combinations that a narrower-coverage training pool (R2+R8)
essentially never sees, so this ceiling won't move without either widening
R2/R8-style training coverage or accepting that the model needs example
coverage of R1's specific vocabulary, not just more readers in the abstract.

## What's next

Milestone 1 continues: rather than adding readers indiscriminately, next
readers should be evaluated on incremental parasha coverage they add to the
pool, not just Gate 2 pass/fail. Milestone 2 (BiLSTM/CRF) should be re-run on
this 3-reader corpus once a stable reader set is settled, since Milestone
2's numbers above are still 2-reader (R1/R2) only.
