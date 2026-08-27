# Ben13 grammar associations, cross-checked against the real corpus

Ben13's combination clips teach fixed "servant → master" ta'am pairings (the
traditional pedagogy for how conjunctive accents attach to disjunctive ones).
This checks whether those taught pairings actually hold up as the dominant
pattern in the real training corpus (`aligned_dataset_merged.json`, same data
every other milestone is measured on), or whether they're just one valid
example among several.

Method: every consecutive within-verse ta'am pair in the corpus is counted;
`P(b|a)` is how often `a` is immediately followed by `b`, out of every time
`a` occurs non-finally. 20 unique pairs appear across Ben13's 15 Torah + 14
Megillot + 14 Haftara combination clips — Megillot and Haftara add no pairs
beyond Torah's set (they use the same base ta'am system minus the
Torah-specific marks), so `[types]` is `teamimtora,teamim-meg,teamim-haf` for
all but one.

## Tier 1 — near-deterministic (Ben13 is teaching the dominant real pattern)

| Pair | count / total | P(b\|a) |
|---|---|---|
| Mahapakh → Pashta | 6,271 / 7,258 | **86.4%** |
| Darga → Tevir | 2,072 / 2,626 | **78.9%** |
| Mercha → Tipecha | 12,965 / 22,209 | **58.4%** |
| Zarqa → Munach | 396 / 734 | **54.0%** |

## Tier 2 — real, but one of several valid continuations

| Pair | count / total | P(b\|a) |
|---|---|---|
| Pashta → Munach | 5,773 / 13,256 | 43.6% |
| Munach → Zaqef_Qatan | 8,822 / 23,057 | 38.3% |
| Qadma → Geresh | 1,786 / 4,707 | 37.9% |
| Munach → Etnachta | 5,589 / 23,057 | 24.2% |
| Tipecha → Mercha | 5,802 / 26,371 | 22.0% |
| Tipecha → Munach | 5,590 / 26,371 | 21.2% |
| Munach → Revia | 3,027 / 23,057 | 13.1% |
| Mercha → Tevir | 1,595 / 22,209 | 7.2% |
| Darga → Munach | 143 / 2,626 | 5.4% |
| Qadma → Munach | 202 / 4,707 | 4.3% |
| Pashta → Mercha | 346 / 13,256 | 2.6% |
| Munach → Segol | 487 / 23,057 | 2.1% |
| Munach → Zarqa | 438 / 23,057 | 1.9% |
| Pashta → Mahapakh | 111 / 13,256 | 0.8% |
| Pashta → Qadma | 71 / 13,256 | 0.5% |

**Munach is a generic servant**, not a 1:1 rule — it precedes at least seven
different disjunctives across the corpus (Zaqef_Qatan, Etnachta, Revia,
Segol, Zarqa among them), each individually real but none dominant. Ben13
teaches each as its own combo clip; none of them should be hard-coded as
*the* continuation for Munach.

## Tier 3 — unvalidatable (real corpus has ~no data to check against)

| Pair | count / total | P(b\|a) |
|---|---|---|
| Yerach_Ben_Yomo → Qarney_Para | 0 / 2 | 0.0% |

Yerach_Ben_Yomo itself occurs only twice in the entire corpus (across all 5
readers), so this "0%" is not a real disagreement with Ben13 — there is
essentially no data to confirm or contradict it. Ben13's clip is the only
place this relationship can be observed at all.

## How this is meant to be used (Milestone 1C)

- **Tier 1** is the candidate set for a CRF transition-matrix prior — nudging
  `LinearChainCRF.transitions` toward these pairs before training, not
  hard-coding them. To be A/B tested against the unbiased CRF on the same
  R1/R2/R8 LORO folds already established for Milestone 2, reported side by
  side.
- **Tier 2** should not bias the CRF — Ben13's example is real but the true
  distribution is multimodal, and forcing it would actively hurt the model's
  ability to represent Munach's genuine variety.
- **Tier 3** has no statistical role here; it only matters for the
  rare-class training-augmentation step (Milestone 1C, step 4), where it's
  the sole source of even a single clean example.

## Result: the CRF prior experiment (Tier 1, step 3)

A/B'd `--crf` vs `--crf --grammar-prior` on the identical R1/R2/R8 LORO
folds and CNN checkpoints (single seed each). Of Tier 1's four pairs, only
Mahapakh→Pashta and Mercha→Tipecha apply — Darga and Zarqa aren't in the
current top-11 class set.

| Fold | CRF (no prior) | CRF + grammar prior | Δ |
|---|---|---|---|
| R1 held out | 57.5% | 58.2% | +0.7 |
| R2 held out | 92.3% | 92.1% | −0.2 |
| R8 held out | 90.7% | 90.5% | −0.2 |
| Mean | 80.17% | 80.27% | **+0.1** |

**Not a real effect at this sample size.** A small gain on the hardest fold
(R1) is cancelled by small losses on the two folds already near ceiling
(R2/R8 both >90%, where there's little room to move and a nudge can only
add noise). Net average moved 0.1 points on a single seed per fold — well
within what a different random init would produce on its own. The grammar
prior is a defensible idea (Tier 1 pairs are real, independently confirmed
patterns) but this implementation, on this corpus, doesn't move the number.
Recorded here rather than left untested or overclaimed, per the project's
standard for small effects (see `results/v3_comparison/FINDINGS.md`).
