# 🎵 Taamify — Deep Learning Classification of Torah Cantillation Marks

Taamify classifies **טעמי המקרא** (Torah cantillation marks) from audio: given a
sung word, the network identifies which of 11 cantillation marks it was sung
with — and a practice app gives learners real-time feedback, like singing-tutor
apps do.

## Headline results (11 classes, chance = 9.1%)

| Benchmark | Accuracy | Rank lift |
|---|---|---|
| Random split (ResNet18 transfer) | 78.0% · MCC 0.753 · AUC 0.972 | — |
| **Unseen reader, same tradition** (mean of both directions) | **51.2%** | +17.3% |
| **Unseen tradition** (mean of both directions) | **19.2%** | +14.2% |

R1 and R2 are Sephardic-Jerusalem; R6 is Ashkenazi. Every cell above is one
reader → one reader, same training cap, same epochs. A model trained on **one**
reader generalizes to a completely different reader of the same tradition at
46–57%. Across the tradition boundary it collapses to 18–20% — in *both*
directions, which differ from each other by only 0.019, so the gap belongs to
the boundary and not to one awkward reader.

**Reader identity is not the barrier; the musical tradition is.** The model
generalizes across voices and fails across nusach.

The grouping is not taken on trust from the video titles: comparing pitch
contours per ta'am, R1 and R2 sit at the within-reader noise floor (0.722 against
a 0.674 floor) while R6 lies well beyond it (1.53–1.57).

But its mistakes keep falling towards the correct rank of the ta'amim
hierarchy, against a baseline that respects the model's own prediction mix:

| | accuracy retained | hierarchy retained |
|---|---|---|
| across the tradition boundary | **32–37%** | **52–82%** |

measured three independent ways (LORO folds, embedding geometry, single-reader
2×2 transfer). The exact ratio moves with the design; the dissociation does not.
**The melodic surface is tradition-specific; the syntactic structure it encodes
is substantially shared.**

> Earlier versions reported 53.0% ± 7.6 for LORO. That number is withdrawn:
> three of its four folds held out readers with a **single recording each**
> (673–1,172 words, all of Bereshit), so they measured noise. See
> `results/v3_comparison/FINDINGS.md`.

A fourth reader (R7, Yemenite) was added to test whether transfer severity
tracks *musical distance*. A noise-control experiment (inject known timing
jitter into R1, watch its metrics decay, solve for the jitter level that
would produce R7's actual numbers) gives **two different answers depending on
the metric**: R7's rank-lift collapse (+0.5%, chance-indistinguishable) is
fully explained by its noisier alignment (~550ms of jitter would do it) and
should not be reported as a finding. Its **accuracy** collapse (9.1%, literal
chance) is not — that same 550ms of jitter predicts ~29% accuracy for a
merely-noisy reader, 20 points above what R7 actually shows.

**So: no claim about Yemenite's *structure* survives scrutiny. A claim that
Yemenite's *melody* transfers even worse than Ashkenazi's — beyond what noise
explains — does.** The two-tradition finding above (R1/R2 vs R6) is the
headline and needs no revision; R7 enters only as an explicitly secondary,
accuracy-only observation. Details in `results/v3_comparison/FINDINGS.md`,
sections 8-9.

## The corpus

```
YouTube (R1, R2 — Sephardic Jerusalem, 82 recordings)
  → mlx-whisper large-v3 → verse-level CTC re-alignment (boundaries from audio)
PocketTorah (R6 — Ashkenazi, 375 aliyot)
  → human-verified word onsets, 99.2% exact token match against the WLC
  → 118,198 labelled words → 128×87 mel-spectrograms → 3 LORO folds
```

Whisper's word boundaries were measured against R6's gold onsets and are off by
a **median −200 ms at the start and −284 ms at the end**, with 53% of starts
wrong by more than 200 ms — they open on the previous word and cut mid-syllable.
Verse-level forced alignment against the canonical text roughly halves that.

## Repository layout

| Path | Purpose |
|---|---|
| `Data_Prep/ingest_pockettorah.py` | PocketTorah → gold-aligned records (reader R6) |
| `Data_Prep/download_readers.py` | re-fetch R1/R2 from YouTube, title → parasha matching |
| `Data_Prep/clean_align_v2.py` | Whisper × Sefaria positional alignment (R1/R2) |
| `Data_Prep/backfill_word_raw.py` | restore the pointed spelling stripped by v2 |
| `Data_Prep/stressed_syllable.py` | stressed syllable from ta'am position; pre/postpositive rules |
| `Data_Prep/ctc_align.py` | per-word CTC alignment → stressed-syllable windows |
| `Data_Prep/verse_align.py` | verse-level CTC → word boundaries from the audio |
| `Data_Prep/validate_verse_align.py` | calibrate and score that alignment against gold |
| `Data_Prep/whisper_vs_gold.py` | measure Whisper's boundary error on chanted Hebrew |
| `Data_Prep/merge_datasets.py` | merge sources, drop readers too small to be a fold |
| `Data_Prep/extract_features_v3.py` | mel-spectrograms, word or stressed window, speaker-norm |
| `00_validate_and_prepare_data.py` | validation + random/LORO splits (`--max-train` to size-match folds) |
| `10_train_pipeline.py` | 10-architecture registry, full metrics, plots, embeddings |
| `11_run_loro.py` | Leave-One-Reader-Out orchestrator |
| `analysis_melisma_spillover.py` | is the ta'am's melody localized on the stressed syllable? |
| `analysis_hierarchy_transfer.py` | does the rank structure survive a change of tradition? |
| `analysis_symmetry.py` | the 2×2 reader→reader transfer: is the tradition gap symmetric? |
| `analysis_tradition_distance.py` | is the tradition grouping acoustic, or just a label? |
| `analysis_embedding_rank.py` | is the embedding geometry organised by rank? (permutation test) |
| `analysis_taam_similarity.py` | embedding distances, dendrogram, traditional-hierarchy test |
| `webapp/` | FastAPI + vanilla JS practice app |
| `results/v3_comparison/FINDINGS.md` | **every measurement in this round, including the negative ones** |

## Quick start (Apple Silicon)

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python 10_train_pipeline.py --model deep_cnn --data-dir prepared_data/random_top11
uvicorn webapp.server:app --port 8077
```

Full rebuild from audio:

```bash
git clone --depth 1 https://github.com/rneiss/PocketTorah.git data/_sources/pockettorah
python Data_Prep/ingest_pockettorah.py
python Data_Prep/download_readers.py            # R1/R2 from YouTube
python Data_Prep/transcribe_mac.py --all-missing
python Data_Prep/clean_align_v2.py
python Data_Prep/merge_datasets.py --require-audio
python Data_Prep/backfill_word_raw.py
python Data_Prep/verse_align.py --readers R1 R2 --all
python Data_Prep/ctc_align.py --dataset aligned_dataset_merged.json --all \
       --write stressed_windows.json
python Data_Prep/extract_features_v3.py --window word
TAAMIFY_FEATURES=$PWD/features_v3_word python 11_run_loro.py \
       --model deep_cnn --holdouts R1 R2 R6 --readers R1 R2 R6 --max-train 40000
```

Run long jobs under `caffeinate -is`. Alignment and feature extraction both
checkpoint per recording and resume.

## Things that were tried and did not work

Kept deliberately — they constrain what the next round should attempt.

- **Windowing on the stressed syllable.** The premise fails twice over: melodic
  activity shows no concentration there for 17 of 21 ta'amim (against a
  randomized-window null), and the whole-word window beats it on every fold and
  every metric. Mahapakh and Yetiv are real exceptions worth revisiting.
- **Montreal Forced Aligner.** No Hebrew acoustic model, dictionary or G2P
  exists. Character-level CTC needs none, because the text is pointed.
- **Mechon Mamre as a fourth reader.** All 187 Torah chapters are downloadable
  at a predictable URL, but the recordings are *spoken, not chanted* — adding
  them would have attached ta'am labels to audio containing no cantillation.
- **Speaker normalization.** Removing each recording's mean and variance in mel
  space costs a little accuracy and moves the cross/within-tradition ratio by
  0.001. The gap is not timbre or channel.

## Data

Raw audio (~12 GB) and derived spectrograms are not in the repo. The alignments
(`aligned_dataset_merged.json`, `stressed_windows.json`,
`verse_aligned_windows.json`) carry a provenance field on every record — `gold`,
`ctc`, `verse_ctc`, `proportional`, `stored` — so any result can be recomputed
on a chosen subset of alignment quality.

## License / citation

Course project (Deep Learning, 2026) by Meir Ben Zion Dvir. Canonical text and
cantillation from the [Sefaria API](https://www.sefaria.org/developers) and the
Westminster Leningrad Codex; audio for reader R6 from
[PocketTorah](https://github.com/rneiss/PocketTorah); transcription by
[Whisper](https://github.com/openai/whisper) (mlx port); forced alignment with
[imvladikon/wav2vec2-xls-r-300m-hebrew](https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew).
