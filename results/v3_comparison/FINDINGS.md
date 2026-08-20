# Taamify v3 — what changed, what was measured, what it means

Everything below was measured on this corpus, with the script that produced it
named. Numbers that are assumptions rather than measurements are labelled as
such. Negative results are kept: three of the ideas that motivated this round
turned out to be wrong, and knowing that is most of the value.

---

## 1. The corpus was rebuilt

| | before | after |
|---|---|---|
| labelled words | 69,798 | **118,198** |
| readers with real coverage | 2 (+3 with one recording each) | **3** |
| meaningful LORO folds | 1 | **3** |
| word boundaries | Whisper, everywhere | gold for 57%, acoustic for 30% |

### The old LORO number was not measurable

The published 53.0% ± 7.6 came from four folds, but R3, R4 and R5 held
**one recording each** — all of Bereshit, 673–1,172 words. Three of the four
folds were noise reported as a generalization score. They are dropped;
`Data_Prep/merge_datasets.py` refuses any reader with fewer than 5 recordings.

### PocketTorah supplies a third reader with gold alignment

`github.com/rneiss/PocketTorah` ships human-verified word onsets next to its
audio (`data/torah/labels/<Parsha>-<n>.txt`). Aligned against the Westminster
Leningrad Codex it bundles — *not* Sefaria, whose maqaf joins differ on a few
dozen verses — the token counts match exactly for **375 of 378 aliyot (99.2%)**,
giving **67,007 words with no alignment error at all**.

Script: `Data_Prep/ingest_pockettorah.py`. Haftarot are excluded: Prophets
cantillation is a different melodic system for the same marks.

---

## 2. Whisper's word boundaries are unusable — measured, not assumed

R6's gold onsets make Whisper's error measurable on chanted Hebrew for the first
time. Same model and settings as the pipeline that produced R1/R2
(`whisper-large-v3-mlx`, `word_timestamps=True`):

| | median | \|err\| p50 | p90 | >200 ms |
|---|---|---|---|---|
| word start | **−200 ms** | 214 ms | 570 ms | **53.1%** |
| word end | **−284 ms** | 288 ms | 764 ms | **68.7%** |

Median gold word: 1,021 ms. The bias is **systematic and negative** — windows
open on the tail of the previous word and close mid-syllable, which is exactly
what the clips sound like. Whisper also recognised only 66.2% of words.

Script: `Data_Prep/whisper_vs_gold.py`.

### Verse-level CTC roughly halves it

Forcing the *canonical verse* (not the surviving records — a third were dropped,
and aligning only survivors would stretch them over audio containing the rest)
onto the audio with `imvladikon/wav2vec2-xls-r-300m-hebrew`:

| vs gold | Whisper | verse CTC | + lag correction |
|---|---|---|---|
| \|start\| p50 | 214 ms | 186 ms | **114 ms** |
| \|end\| p50 | 288 ms | 170 ms | **161 ms** |
| start >200 ms | 53.1% | — | **26.8%** |

CTC emits a character only after accumulating evidence, so its onsets lag by a
near-constant +170 ms; that part is calibration, not noise.

**Assumption, not measurement:** the lag constants were fitted on R6. R1/R2 have
no gold to check against, and their Whisper bias behaves differently (+27 ms
disagreement rather than −200 ms). What is established is that the method halves
the error where truth exists.

Scripts: `Data_Prep/verse_align.py`, `Data_Prep/validate_verse_align.py`.
Result: 35,958 of R1/R2's words re-derived from audio; 13,165 rejected because
the surviving text did not span the verse, which would have made CTC compress a
full verse into a partial window.

---

## 3. MFA has no Hebrew — and vocalized text makes that irrelevant

67 acoustic models, 53 dictionaries, 51 G2P models in the official repository;
Arabic yes, Hebrew no. The reason is that unvocalized modern Hebrew makes G2P
near-impossible — which does not apply here, because the Masoretic text is fully
pointed. A character-level CTC model needs no dictionary at all: the Hebrew
wav2vec2 vocabulary is exactly the 27 letters plus a word delimiter.

---

## 4. Two real bugs in the phonology

`Data_Prep/stressed_syllable.py` derives the stressed syllable from the position
of the ta'am inside the pointed spelling, handling the postpositive
(Segolta, Zarqa, Pashta, Telisha Qetana) and prepositive (Telisha Gedola, Yetiv,
Dehi) families, whose marks sit at a word edge rather than on the stress. The
Masoretes *double* those marks when the stress falls elsewhere, so a doubled
Pashta locates the stress — verified on real words (`הַשָּׁמַ֙יִם֙`, `הָאָ֙רֶץ֙`).

Two errors were found by testing against hard cases rather than by inspection:

1. **Furtive patah** — `רוּחַ` counted as two syllables. The final patah under a
   guttural is an off-glide; it opens no syllable and never carries stress.
2. **Shuruq and holam male** — `יִשְׁמְעוּ` counted as two. `וּ` is a vowel encoded
   as vav+dagesh, and dagesh was not in the vowel set, so **every shuruq and
   holam male in the corpus was invisible**.

Effect: the corpus moved from 75.9% to **81.4% milra**; ~3,700 words had the
wrong stressed syllable, hence the wrong audio window. All 14 test cases pass —
including dagesh chazak, segolates, qamats qatan and consecutive shevas.

Every alignment computed before the fix was discarded and recomputed.

---

## 5. The stressed-syllable window was a bad idea — twice measured

This was the headline methodological proposal of the plan. It is wrong.

### Acoustically: no melodic concentration

Melodic activity `a(t) = |d/dt semitones(f0(t))|`, in semitones relative to each
reader's median, against a null where an identically sized window is placed at
random inside the word (`analysis_melisma_spillover.py`, 37k words, R6 only —
gold boundaries):

| ta'am | n | melody in syllable | syllable width | null | lift |
|---|---|---|---|---|---|
| Pazer | 27 | 82.0% | 77.0% | 82.6% | −0.6% |
| Zarqa | 175 | 71.9% | 70.8% | 74.8% | −2.9% |
| Tipecha | 6,734 | 54.1% | 47.5% | 54.0% | +0.1% |
| Etnachta | 3,351 | 46.5% | 48.7% | 55.1% | −8.6% |
| **Mahapakh** | 1,778 | 61.3% | 43.7% | 46.6% | **+14.7%** |
| **Yetiv** | 199 | 74.3% | 50.9% | 61.0% | **+13.3%** |

The lift is zero or negative for **17 of 21 ta'amim**: melody is spread across
the word in proportion to duration. Energy (RMS) was deliberately not used —
loudness peaks on the stressed syllable by definition, so it would have
confirmed the hypothesis for every ta'am regardless of melody.

The long melismas that motivated the hypothesis show no spillover. Pazer
explains itself: its stressed syllable already occupies **77% of the word**, so
there is nothing to spill.

**Mahapakh and Yetiv are genuine exceptions** — both concentrate melody in a
narrow syllable. Worth following up on other readers.

### In classification: the whole word wins

Same records, same labels, same folds, one variable:

| metric | word | stressed |
|---|---|---|
| accuracy | **0.3785 ± 0.152** | 0.3199 ± 0.129 |
| balanced accuracy | **0.3878** | 0.3201 |
| MCC | **0.3136** | 0.2505 |
| AUC macro | **0.7857** | 0.7366 |

Word wins on every fold and every metric. Narrowing to the stressed syllable
removes information, not noise. The prediction was stated before the run.

**The ta'am is realized across the word, not localized on the stressed
syllable.** Two independent methods, same conclusion.

---

## 6. The finding that reframes the project

LORO on the three readers, word window:

| metric | R1 | R2 | **R6** |
|---|---|---|---|
| accuracy | 0.462 | 0.509 | **0.165** |
| MCC | 0.397 | 0.457 | **0.086** |
| AUC macro | 0.832 | 0.894 | **0.632** |

R1 and R2 are both **Sephardic Jerusalem**; R6 is **Ashkenazi**. Holding out R1
or R2 still leaves the other tradition-mate in training. Only the R6 fold tests
a tradition the model has never heard — and it collapses to near chance (9.1%).

The claim is sharper than "the model learned readers": **the model learns a
nusach, not the cantillation system.**

**Confound, controlled and refuted.** The R6 fold also trains on 45% less data
(51,191 vs 91,167/94,038), so the collapse could have been starvation rather
than a foreign tradition. Capping every fold to the same 40,000 training samples
(`--max-train`, stratified so class balance is untouched):

| fold | tradition | uncapped | **capped to 40k** |
|---|---|---|---|
| R1 | Sephardic-Jerusalem | 0.462 | **0.472** |
| R6 | Ashkenazi | 0.165 | **0.153** |

R1 is unchanged — it never needed the extra data. R6 is unchanged too. **At
identical training size the gap is still threefold.** Data starvation is not the
explanation; the musical tradition is.

### The hierarchy transfers better than the melody does

Errors staying inside the same traditional rank, size-matched folds, against a
baseline that respects the model's own prediction mix:

| fold | tradition | accuracy | within-rank | chance | lift | 95% CI |
|---|---|---|---|---|---|---|
| R1 | Sephardic-Jerusalem | 0.472 | 38.0% | 21.8% | **+16.2%** | [+15.6, +16.9] |
| R6 | Ashkenazi (unseen) | 0.153 | 30.3% | 21.9% | **+8.4%** | [+8.1, +8.7] |

**Accuracy retains 32% across the tradition boundary. The rank structure retains
52%.** The model that can barely name an Ashkenazi ta'am still errs towards the
right rank at well above chance.

This is the thesis, quantified: the melodic surface is tradition-specific, the
syntactic structure it encodes is substantially shared.

Per rank, in the cross-tradition fold — with chance computed *per rank*, since a
crowded rank catches stray predictions by size alone and a single global
baseline would flatter it:

| true rank | errors | within rank | chance | lift |
|---|---|---|---|---|
| Emperor (קיסרים) | 8,215 | 19.0% | 11.2% | **+7.7%** |
| King (מלכים) | 19,628 | 9.6% | 4.9% | **+4.7%** |
| Duke (משנים) | 6,467 | 8.3% | 5.9% | +2.4% |
| Servant (משרתים) | 18,339 | 65.4% | 50.5% | **+14.8%** |

Every rank is positive, so the aggregate effect is not an artefact of the
conjunctives being numerous — though they carry it most strongly. The
conjunctive/disjunctive split, the most fundamental division in the system,
is what survives best.

### The gap is symmetric, and it is not about readers

The LORO result was measured one way round, which leaves a duller explanation
open: perhaps R6 is simply harder audio. Single-reader transfers settle it —
each trained on one reader, tested on one reader, capped to 30,000 samples
(`analysis_symmetry.py`):

The full 2×2, every cell one reader → one reader, same cap, same epochs:

| train on | test on | | accuracy | rank lift | 95% CI |
|---|---|---|---|---|---|
| R1 (Sephardic-Jm) | R2 (Sephardic-Jm) | within | **0.565** | +19.6% | [+19.0, +20.3] |
| R2 (Sephardic-Jm) | R1 (Sephardic-Jm) | within | **0.459** | +14.9% | [+14.2, +15.7] |
| R6 (Ashkenazi) | R1 (Sephardic-Jm) | cross | **0.182** | +11.4% | [+10.8, +11.9] |
| R6 (Ashkenazi) | R2 (Sephardic-Jm) | cross | **0.202** | +17.1% | [+16.5, +17.7] |
| | | **within** | **0.512** | **+17.3%** | |
| | | **cross** | **0.192** | **+14.2%** | |

**One reader generalizes to a completely different reader of the same tradition
at 46–57%.** Across the tradition boundary, in *both* directions, it collapses
to 18–20% against a 9.1% chance — and **the two cross-tradition directions
differ by only 0.019**, so the gap belongs to the boundary and not to one
awkward reader.

#### Over three seeds

Every other number in this document is a single run, which is a point estimate
wearing the clothes of a measurement. The comparison the whole claim rests on
was repeated under three initializations (`TAAMIFY_SEED`):

| transfer | seed 42 | seed 7 | seed 123 | mean ± std |
|---|---|---|---|---|
| R1 → R2 (within) | 0.5654 | 0.5561 | 0.5342 | **0.552 ± 0.013** |
| R6 → R2 (cross) | 0.2016 | 0.2003 | 0.2178 | **0.207 ± 0.008** |

The distributions do not come close to touching. The gap is not seed luck.

So reader identity is not the barrier. The earlier reading — "the model learned
to recognise readers" — is withdrawn. The model generalizes across voices
perfectly well; it does not generalize across nusach.

A side observation worth following up: training on R1 alone beats training on
R1+R6 when testing Sephardic material (0.565 vs 0.472). Adding the Ashkenazi
reader appears to *interfere* rather than help, which is what one would expect
if the two traditions assign different melodies to the same labels.

### The grouping is acoustic, not a label I read off the titles

Everything above rests on R1 and R2 belonging to one tradition and R6 to another
— and that came from the YouTube titles ("נוסח ספרדי ירושלמי") plus PocketTorah
being an American Ashkenazi recording. A label, not a measurement, and if R1 and
R2 were as different from each other as either is from R6 the framing collapses.

Measured (`analysis_tradition_distance.py`): per ta'am and reader, the pitch
contour of each instance in semitones relative to that reader's median, warped
to a common grid and averaged; the distance between readers is the mean distance
between their average contours over 20 shared ta'amim. Pitch rather than
spectrum, because timbre is exactly what must *not* be compared — two people can
have very different voices and sing the same tune. The floor comes from splitting
one reader's own instances into two halves.

| pair | contour distance | vs floor |
|---|---|---|
| R1 vs itself (half/half) | 0.530 | — |
| R2 vs itself | 0.589 | — |
| R6 vs itself | 0.904 | — |
| **R1 vs R2** | **0.722** | floor **+0.048** |
| R1 vs R6 | 1.569 | floor +0.895 |
| R2 vs R6 | 1.531 | floor +0.857 |

**R1 and R2 sit essentially on the noise floor** — their melodic contours for the
same ta'amim are close to indistinguishable. R6 is far outside it. The grouping
is real.

Two honest caveats. The script prints an "18.4×" ratio, which is
`0.876 / 0.048` — a division by something at noise level, so it is arithmetically
correct and rhetorically inflated. The defensible statement is that R1↔R2 lies at
the floor while R6 lies well beyond it. And R6 is *internally* more variable
(0.904 against 0.530/0.589), so part of its distance is its own spread — though
1.55 still clearly exceeds its own 0.90.

### Speaker normalization does not touch it

If the gap were timbre, channel or room — anything about *how* a voice sounds
rather than *what* it sings — subtracting each recording's own mean and variance
in mel space should have narrowed it. Matched transfers, same cap, same epochs:

| transfer | raw | + speaker-norm |
|---|---|---|
| R1 → R2 (within tradition) | 0.5654 | 0.5443 |
| R6 → R2 (cross tradition) | 0.2016 | 0.1936 |
| **cross / within ratio** | **0.357** | **0.356** |

Normalization costs a little accuracy in both directions and moves the ratio by
0.001. **The tradition gap is not a speaker or channel effect.** It sits in the
melodic content itself — which is what the hypothesis predicts and what a
timbre-based explanation cannot survive.

### Three designs, one dissociation

| design | accuracy retained | hierarchy retained |
|---|---|---|
| LORO folds, size-matched | 32% | 52% |
| Embedding geometry (rank gap) | — | 59% |
| Single-reader transfer, full 2×2 | 37% | 82% |

The exact ratio depends on the design — the 2×2 gives the highest figure partly
because its within-tradition baseline is lower (R2 → R1 lifts only +14.9%). What
is stable across all three, and is the claim worth making, is the **dissociation
itself**:

> Accuracy retains roughly a third across the tradition boundary. The
> hierarchical structure retains between half and four fifths. In every design,
> by every statistic, the structure survives substantially better than the
> ability to name the mark.

Reporting the range rather than the friendliest number: **32–37% accuracy,
52–82% hierarchy.**

Script: `analysis_hierarchy_transfer.py`, with a chance baseline that respects
the model's own prediction distribution, so over-predicting a crowded rank
cannot manufacture the effect.

### The geometry is rank-organised too — independently of the errors

The error analysis could in principle be an artefact of which classes are easy.
The embedding space is a separate test (`analysis_embedding_rank.py`,
size-matched folds, 3000 permutations):

| fold | rank gap | perm p | silhouette by **rank** | silhouette by **ta'am** |
|---|---|---|---|---|
| R1 — within tradition | **+0.0700** | 0.025 | **+0.090** | −0.018 |
| R6 — cross tradition | **+0.0414** | **0.005** | −0.012 | −0.092 |

In both folds the ranks separate *better than the individual marks do*. A
negative silhouette on the ta'amim means the eleven classes overlap heavily,
while the coarse traditional grouping still coheres. The permutation shuffles
rank membership, so an effect driven by which marks happen to share a rank does
not count.

**The rank gap retains 59% of its size across the tradition boundary** — against
32% for accuracy, and closely matching the 52% the error analysis gave
independently. Two unrelated measures, one on mistakes and one on geometry,
agree on how much of the hierarchy survives when the melody does not.

Three independent lines — error direction, centroid geometry, cluster
separation — point the same way.

---

## 7. Provenance is a column, not an assumption

Every feature row records which alignment produced it (`gold`, `ctc`,
`verse_ctc`, `proportional`, `stored`). This caught a real contamination: 2,516
rows in the word-window set had survived in stale checkpoints from an earlier
smoke test and still carried Whisper boundaries. They were labelled `word` while
everything else said `verse_ctc`, which is the only reason it was visible.

It also caught `results/loro_R2_top11` holding a `resnet18_tl` run from 19 July,
which `11_run_loro.py`'s `glob('*/*_metrics.json')` would have aggregated into
the new table as if it belonged. Archived to `results/_archive_pre_v3/`.

Neither failure would have crashed anything. Both would have produced a
plausible number.

---

## 8. The R7 noise-control result: the Yemenite collapse is not yet a finding

Section 6 raised an open question: R7's near-total collapse on both accuracy
(0.091, literal chance) and rank lift (+0.5%, CI touching zero) could mean
Yemenite tradition is genuinely more structurally distant, or could mean R7's
measurably noisier alignment (median CTC score -2.79 against R1's -1.91) is
attenuating a real effect toward zero — noisy labels shrink observed effects
mechanically, which is arithmetic, not a hypothesis.

The control: inject Gaussian timing noise, independently on each window's start
and end, into R1 alone (`--jitter-ms`, `Data_Prep/extract_features_v3.py`) —
same architecture, same training budget, same evaluation, only the boundary
precision degraded — and watch where R1's own rank lift goes.

| jitter (ms, sd) | accuracy | rank lift | 95% CI |
|---|---|---|---|
| 0 (clean) | 47.2% | +16.2% | [+15.6%, +16.9%] |
| 100 | 45.6% | +14.9% | [+14.3%, +15.6%] |
| 200 | 42.8% | +12.3% | [+11.6%, +12.9%] |
| 400 | 36.0% | +6.5% | [+5.9%, +7.1%] |

**Extrapolating the curve past 400ms lands on R7's observed +0.5% at roughly
600ms of injected jitter.** That is not an implausible amount of timing error
for R7's pipeline: R1/R2's *corrected* boundaries land at 114-161ms median
error against gold, but that correction (verse-level CTC informed by Whisper's
macro-location) is exactly what failed for R7 — its accepted windows come from
the proportional fallback, whose confidence gap versus R1 (sd 0.29 vs 0.56,
median score -2.79 vs -1.91) is entirely consistent with timing error in the
few-hundred-millisecond range implied by this curve.

**Conclusion: R7's collapse cannot currently be reported as evidence that
Yemenite tradition is structurally more distant.** Noise alone plausibly
accounts for all of it. The two-tradition finding (Section 6, R1/R2 vs R6)
stands on its own — it passed four independent confound checks on comparably
clean data. The three-tradition distance-gradient story does not yet stand,
and should not be claimed until R7 has an alignment pipeline whose confidence
is closer to R1/R2/R6's, or an independent, cleaner source for a third
tradition replaces it.

Script: `Data_Prep/extract_features_v3.py --jitter-ms --jitter-readers`.

### Tried to fix it directly — didn't work

If R7's problem is timing precision, not label correctness (confirmed —
`proportional_verse_locate.py`'s ta'am sequences were hand-verified against
known verses before being trusted), a second forced-alignment pass over a
*narrow* window around each already-correctly-labeled word should tighten
timing the same way it did for R1/R2/R6's per-word pass in `ctc_align.py`.

Tried at two window widths (`Data_Prep/refine_r7_alignment.py`, pilot on one
recording): 0.45s and 1.0s padding around the pass-one span. Both made scores
*worse*, not better — mean -4.08 and -3.44 respectively, against pass-one's
-2.79 and R1's -1.91. Isolating a single word loses the sentence-level context
the verse-level pass had to constrain the search; that loss outweighs the gain
from a narrower window. **This is not a search-space problem** — it looks like
a genuine mismatch between `imvladikon/wav2vec2-xls-r-300m-hebrew` and
Yemenite phonology that a smaller window can't route around. The corpus was
restored from backup after both trials; this result is not reflected in
`aligned_dataset_r7.json`.

A third variant was tried before concluding this: keep the *verse-level*
context (don't isolate single words) but tighten the macro-window to pass-one's
own word-boundary bounding box instead of the original proportional-character
estimate. Result: median -2.85 against pass-one's -2.79 — statistically a
wash, not an improvement. Window width, at either the word or the verse level,
is not the lever. Three attempts, three negative results, converging on the
same conclusion: this is the acoustic model's unfamiliarity with Yemenite
phonemes, not a search-space problem.

What's left: a Hebrew acoustic model actually trained on Yemenite or Mizrahi
pronunciation (none found), or a cleaner source recording — see `outreach/`.

### Accuracy and rank-lift disagree about how much noise explains

If R7 were simply "R1, but noisier," one noise level should explain both of
its collapsed metrics. Fitting each curve separately and solving for R7's
actual observed value says otherwise:

| target | jitter needed |
|---|---|
| rank lift -> R7's +0.5% | ~550 ms |
| accuracy -> R7's 9.1% | ~873 ms |

**A single noise level cannot account for both.** At the rank-lift-implied
noise level (550ms), R1's accuracy curve predicts ~29% — R7's actual accuracy
is 9.1%, a 20-point gap noise does not cover.

Read together with section 8: **the rank-lift collapse is fully explained by
noise and should not be claimed as a finding. The accuracy collapse is not** —
something beyond timing noise is also driving it down, consistent with (though
not proof of) a genuine tradition-distance effect on top of the alignment
problem. The two metrics get different verdicts, not the same one.

**Reporting consequence:** R7 does not belong in the headline claim (Section
6, R1/R2 vs R6, which stands alone and needs no revision). It belongs as an
explicitly secondary, accuracy-only observation: Yemenite transfer collapses
further than Ashkenazi's, by more than its alignment noise alone can explain
— suggestive of a distance gradient, not confirmation of one. Any claim about
its *hierarchical structure* specifically is unsupported and should not be
made.

## 8. Open

- **A fourth reader, genuinely chanted.** Mechon Mamre has all 187 Torah
  chapters at a predictable URL, but its own terms say the recordings are
  "not chanted with a melody" — spoken Sephardic-style Hebrew, licensed from a
  third party. Adding it would have poisoned the labels silently. Teaching
  recordings from known ba'alei kri'ah, with written permission, remain the
  realistic path.
- **Boundary-quality ablation** — drop the rows still marked `stored` (Whisper
  boundaries that verse alignment could not replace) and re-measure. Built,
  pending. Speaker normalization is done and reported above: it does not help.
- **CRF over the sequence model** — `14_sequence_model.py` has a BiLSTM and no
  transition constraints. The corpus supports it: of 5,785 verses, 5,581 have
  exactly one silluq and 5,424 exactly one etnachta, with **none** having two.
