# Milestone 4 — widen past top-11, and Focal Loss vs. weighted CE

Widened LORO (R1/R2/R8, deep_cnn) from the established 11 classes to 21 —
every class with real training volume (Tipecha down to Pazer, 138+
instances); the four essentially-unusable classes (Mercha_Kefula=12,
Yerach_Ben_Yomo=2, Shalshelet=2, Qarney_Para=0) are excluded here and left
to Milestone 1C's `BEN13AUG` prep. Same LORO discipline as every other
milestone: `--num-classes 21`, `--max-train 40000`, same three folds.

Also revisited Focal Loss (proposed early in the project, never adopted,
never tested head-to-head against weighted CE) — `10_train_pipeline.py`
gained a self-contained `FocalLoss` class and `--loss {ce,focal}` flag,
since severe class imbalance (Tipecha:Pazer ≈ 96:1 at 21 classes) is exactly
where Focal Loss's case is supposed to be strongest.

## Result: 21-class LORO, weighted CE vs. Focal Loss

| Fold | CE accuracy | CE balanced | Focal accuracy | Focal balanced |
|---|---|---|---|---|
| R1 held out | 41.7% | 36.9% | 33.1% | 36.1% |
| R2 held out | 70.8% | 74.3% | 48.9% | 68.0% |
| R8 held out | 66.0% | 70.3% | 51.2% | 66.0% |
| **Mean** | **59.5%** | **60.4%** | **44.4%** | **56.7%** |

**Focal Loss loses on every fold, on both metrics, with no exceptions** —
a clean, consistent negative result, not a mixed one. The gap is largest on
raw accuracy (-15.1 points mean) and smaller but still real on balanced
accuracy (-3.8 points mean), meaning Focal Loss isn't even delivering on
its own specific promise (better minority-class treatment) enough to
justify its cost to overall accuracy here. **Weighted CE stays the loss
function for this pipeline.** Single seed per fold — not multi-seeded, so
treat the magnitude as indicative, but the direction is unambiguous across
all three independent folds.

## Widening itself costs accuracy, as expected

Going from 11 to 21 classes drops accuracy on every fold versus the
Milestone 0/1 baseline (e.g. R1: 46.9% → 41.7%; R2: 82.7% → 70.8%), because
several of the newly-added classes generalize poorly to an unseen reader —
some previously-strong top-11 classes also degrade from the added confusion
(e.g. R1's Tevir, solid at 11 classes, drops to F1 0.07 at 21). This is the
expected, known cost of widening flagged in the plan going in — not a
regression to chase, a baseline for what widening costs before any
mitigation (sequence context, more readers, or Milestone 1C's rare-class
prep) is applied on top of it.

## A real infrastructure fix, found along the way

`10_train_pipeline.py`'s training loop was calling `.cpu().numpy()` on
predictions inside every batch (625 batches/epoch at this dataset size) —
on MPS this forces a blocking GPU→CPU sync on every batch, serializing what
should be async dispatch. Fixed to accumulate predictions on-device and
sync once per epoch instead (numerically identical results, confirmed by
construction — same predictions, same order, just batched into one
transfer). Measured effect was smaller than expected (~27%, comparing
same-epoch-count runs) because an unrelated stuck process from a different
project (`Blog/embed_index.py`, running 2.5 days in uninterruptible sleep)
was holding the system's swap nearly full throughout these runs, masking
most of the fix's real benefit. Worth re-measuring cleanly once that's
resolved.
