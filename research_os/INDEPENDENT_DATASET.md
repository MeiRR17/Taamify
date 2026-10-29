# Independent BEN13 dataset preparation

Run from the repository root:

```bash
OMP_NUM_THREADS=1 venv/bin/python -m research_os.prepare_independent_dataset \
  --out prepared_data/ben13_source_independent_NEW_NAME

OMP_NUM_THREADS=1 venv/bin/python -m unittest research_os.test_independent_dataset -v
```

The output directory must not exist. This command does not modify legacy
preparations, train a model, or change the application's checkpoint.

## Saved contract

- `train.npz`, `val.npz`, `calibration.npz`, `test.npz` each contain `X`, `y`,
  `sample_ids`, and original `source_rows`. X/y retain the existing training
  pipeline's shapes, class order and per-sample normalization.
- Every split has a row-aligned JSONL manifest. It records canonical-text label
  provenance, source row, pseudo speaker ID, original verse/block, feature hash,
  connected component, alignment quality and boundaries. Original session ID is
  unknown and remains null.
- `metadata.json` records source-file hashes, exact builder hash, feature contract,
  fixed heldout parashot, seed, class support, output-file hashes, exclusions,
  assignment fingerprint and limitations. Read it only when `status=ready` and
  neither `BUILDING` nor `BUILD_FAILED` exists.
- `excluded.jsonl` accounts for rejected and quarantined rows. They are not
  silently relabeled or counted as successes.

All words from a verse are kept together, conservatively also linking source
blocks and exact normalized-feature duplicates. Components that bridge the
canonical test boundary are quarantined. Conflicting canonical labels for
identical model inputs quarantine their connected components. Development and
calibration target 10% each of the remaining non-test rows; indivisible groups
mean exact percentages are not guaranteed. Rare-class groups are reserved for
training first; missing development/calibration classes are reported explicitly.

Consumers should validate the contract before training:

```python
from pathlib import Path
from research_os.prepare_independent_dataset import verify_dataset
metadata = verify_dataset(Path('prepared_data/ben13_source_independent_20261028_v1'))
```

This verifies saved file hashes, row IDs/labels against NPZ arrays, assignment
fingerprint and zero overlap by sample, source block, verse, feature and component.

## First actual build — 28 October 2026

`prepared_data/ben13_source_independent_20261028_v1` is ready:

| Split | Rows |
|---|---:|
| Training | 40,318 |
| Development (`val`) | 5,047 |
| Calibration | 5,040 |
| Historical heldout test | 10,189 |

All 21 declared classes occur in every split. All six split pairs have zero
overlap on sample ID, source block, verse, exact normalized feature and connected
component. Dataset fingerprint:
`6dfa79cb3197c2022e2b7427912e166106eabb708d1974b07720e0f3ed7b99c7`.

The previous feature quality rules exclude 1,998 too-silent rows, 1,447 rows with
too much padding and 11 out-of-scope class rows. An additional 599 rows belonging
to components with conflicting exact-feature labels are quarantined. This
includes 112 former test rows, so the new 10,189-row test and the old 10,301-row
test cannot be used as an unqualified before/after accuracy comparison.

The first builder was captured in a local reproduction snapshot at
`results/research_os/data_preparation/ben13_source_independent_20261028_v1/executed_builder.py`.
Its hash matches the immutable dataset metadata. The subsequently added consumer
verifier does not alter that existing dataset. Verification details remain local
in the same results directory.

## Limits that downstream work must preserve

This preparation fixes recording/feature split independence; it does not prove
alignment correctness, sound-label correctness, child/new-speaker performance,
or fresh final-test independence. Labels still originate in the expected text.
Low-alignment flags and suspicious durations remain visible for separate
ablation/review rather than being silently filtered to inflate a score. Exact
feature duplicate checks are not acoustic near-duplicate detection. The
historical canonical test has already been inspected many times.

An existing checkpoint may have trained on the new development/calibration rows
under its historical split. Its score on them is therefore a diagnostic replay,
not a fair generalization baseline against a freshly trained model. Every
dependent learned component must use the new manifests, not draw its own split.

## Bounded development pilot

```bash
OMP_NUM_THREADS=1 venv/bin/python -m research_os.train_independent_candidate \
  --data-dir prepared_data/ben13_source_independent_20261028_v1 \
  --out results/research_os/training/independent_cnn_NEW_NAME \
  --device mps --max-epochs 5 --max-seconds 600

OMP_NUM_THREADS=1 venv/bin/python -m unittest research_os.test_independent_training -v
```

This runner initializes the existing DeepCNN architecture freshly with seed
20261028, balanced cross-entropy, Adam and no augmentation. It decodes only train
and development arrays; test/calibration artifacts are verified as opaque hashes
without reading their labels/features through NumPy. It chooses the best epoch
using development macro-F1 and retains per-class precision/recall, confusion,
epoch times and process peak RSS. No production swap or test-score reporting is
performed.

`last.pth` supports resuming the same run with `--resume <run>/last.pth`; a changed
dataset, class order, seed/recipe or source hash is rejected. Total epochs and
runtime remain bounded across resumes. The saved batch cursor and deterministic
per-epoch permutation support resuming a partial epoch. Do not change runner
source mid-run. MPS is a single shared device: coordinate this pilot with other
heavy inference/training work.
