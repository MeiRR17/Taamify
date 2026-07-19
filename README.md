# 🎵 Taamify — Deep Learning Classification of Torah Cantillation Marks

Taamify classifies **טעמי המקרא** (Torah cantillation marks) from audio: given a
sung word, the network identifies which of 11 cantillation marks it was sung
with — and a practice app gives learners real-time feedback, like singing-tutor
apps do.

**Headline results** (11 classes, chance = 9.1%):

| Benchmark | Model | Result |
|---|---|---|
| Random split (test) | ResNet18 Transfer Learning | **78.0% acc, 79.9% balanced, MCC 0.753, AUC 0.972** |
| Unseen-reader (LORO, 4 folds) | **DeepCNN** | **53.0% ± 7.6** (beats ResNet18's 51.4% — small model generalizes better) |
| + verse-context BiLSTM | frozen CNN + tagger | **+7.9 accuracy points** from ta'amim grammar |

Scientific finding: 49.2% of the model's errors stay within the same
*traditional* rank of the centuries-old ta'amim hierarchy (1.50× chance) —
a network trained on audio alone rediscovers the classical taxonomy.

## The pipeline

```
YouTube readings (5 readers, 111 recordings, ~60h)
  → mlx-whisper large-v3 (word-level timestamps)
  → precision-first cleaning against Sefaria's cantillated canonical text
    (verse-exact parasha bounds, positional alignment, exact matches only)
  → 69,798 verified labeled words → 64,517 mel-spectrograms (128×87)
  → 10 architectures compared → LORO generalization test → analyses
```

## Repository layout

| Path | Purpose |
|---|---|
| `Data_Prep/` | transcribe (`transcribe_mac.py`), clean+align (`clean_align_v2.py`), extract features (`extract_features_v2.py`) |
| `00_validate_and_prepare_data.py` | validation + random/LORO splits → `prepared_data/` |
| `10_train_pipeline.py` | 10-architecture registry, full metrics, plots, embeddings |
| `11_run_loro.py` | Leave-One-Reader-Out orchestrator |
| `13_hyperparam_sweep.py` | lr × batch grid + regularization ablations |
| `14_sequence_model.py` | verse-level BiLSTM over frozen CNN embeddings |
| `analysis_taam_similarity.py` | embedding distances, dendrogram, traditional-hierarchy test |
| `12_gradcam.py` | Grad-CAM explainability |
| `webapp/` | Web application (FastAPI + vanilla JS): practice mode with golden-teacher clips and pitch DTW, recording analysis |
| `report/Taamify_Report.md` | full project report (Hebrew) |
| `legacy/` | archived v1 artifacts (superseded; safe to delete) |

## Quick start (Apple Silicon)

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# with prepared data in place:
python 10_train_pipeline.py --model deep_cnn --data-dir prepared_data/random_top11
uvicorn webapp.server:app --port 8077
```

Full-pipeline reproduction (needs the audio in `data/R*/wav/`):

```bash
python Data_Prep/transcribe_mac.py --all-missing
python Data_Prep/clean_align_v2.py
python Data_Prep/extract_features_v2.py
python 00_validate_and_prepare_data.py --split random
python 10_train_pipeline.py --model all
python 11_run_loro.py --model resnet18_tl,deep_cnn
```

Practical notes: run long jobs with `caffeinate -is` (macOS sleep kills them);
if the project lives in a cloud-synced folder, set `TAAMIFY_FEATURES` to a
local copy of `features_v2/` — cloud eviction corrupts long numpy reads.

## The app

`uvicorn webapp.server:app --port 8077`, then open http://localhost:8077 — three views:
1. **Practice** — pick a ta'am, hear a real teacher clip from the Torah, sing
   it back; scored 70% by the CNN's probability + 30% by a pitch-curve DTW
   (pyin F0, key-normalized), drawn over the teacher's melodic "tube".
2. **Recording analysis** — Whisper segments any recording; every word gets a
   top-3 ta'am prediction.
3. **About** — model card and provenance. Recording happens in the browser
   (raw PCM capture over the Web Audio API; no plugins, no ffmpeg).

## Data

Raw audio (~15 GB) and derived spectrograms are not in the repo. The verified
alignment (`aligned_dataset_v2.json`, `metadata_cleaned_v2/`) and the audit
report (`data_audit_report.md`) are included; features are regenerable with
one command from the audio.

## License / citation

Course project (Deep Learning, 2026) by Meir Ben Zion Dvir. Canonical text and
cantillation from the [Sefaria API](https://www.sefaria.org/developers);
transcription by [Whisper](https://github.com/openai/whisper) (mlx port).
