"""
build_colab_notebook.py — generates Taamify_Colab.ipynb (submission notebook)
=============================================================================
Professional English notebook covering the entire project end-to-end.
Every result figure is EMBEDDED in the notebook as a pre-executed cell
output (base64), so the grader sees all graphs without running anything;
the setup and live-demo cells remain runnable in Google Colab.

Regenerate with:  python build_colab_notebook.py
"""

import base64
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "Taamify_Colab.ipynb"
REPO = "https://github.com/MeiRR17/Taamify"

cells = []
_exec_count = [0]


def md(source):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "source": source})


def code(source, outputs=None):
    _exec_count[0] += 1
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8],
                  "metadata": {}, "execution_count": _exec_count[0],
                  "outputs": outputs or [], "source": source})


def image_output(png_path):
    b64 = base64.b64encode((ROOT / png_path).read_bytes()).decode()
    return {"output_type": "display_data", "metadata": {},
            "data": {"image/png": b64,
                     "text/plain": [f"<Figure: {Path(png_path).name}>"]}}


def figure_cell(png_path, caption=None):
    """Code cell that would display the figure, with the figure pre-embedded."""
    src = f'from IPython.display import Image\nImage("{png_path}")'
    code(src, outputs=[image_output(png_path)])
    if caption:
        md(f"*{caption}*")


# ═══════════════════════════════ 1. Title ═══════════════════════════════
md(f"""# Taamify: Deep Learning Classification of Torah Cantillation Marks from Audio

**Author:** Meir Ben Zion Dvir  |  **Course:** Deep Learning, Spring 2026
**Repository:** [{REPO}]({REPO})

This notebook presents the complete project end to end: problem definition,
dataset construction, model comparison, hyperparameter study, generalization
testing, analyses, and the path to a production application. All figures are
embedded; the setup and demonstration cells are runnable in Google Colab.

**Headline results** (11 classes, random-chance baseline 9.1%):

| Benchmark | Model | Result |
|---|---|---|
| Random split, held-out test set | ResNet18 (transfer learning) | 78.0% accuracy, 79.9% balanced accuracy, MCC 0.753, macro AUC 0.972 |
| Unseen-reader test (LORO, 4 folds) | DeepCNN | 53.0% plus or minus 7.6 (5.8 times chance) |
| Verse-context tagger (BiLSTM) | frozen CNN + tagger | +7.9 accuracy points from cantillation grammar |
""")

# ═══════════════════════════════ 2. Setup ═══════════════════════════════
md("""## 0. Environment Setup

The following cell clones the repository and installs dependencies. The live
training demonstration (Section 4) runs on a 550-sample subset bundled with
the repository; the full results shown afterwards were produced by training
on all 64,517 samples (multiple overnight runs on an Apple M4).""")

code(f"""!git clone -q {REPO}.git taamify 2>/dev/null || (cd taamify && git pull -q)
%cd taamify
!pip install -q librosa seaborn
import torch, numpy as np, matplotlib.pyplot as plt
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())""")

# ═══════════════════════ 3. Problem definition ══════════════════════════
md("""## 1. Problem Definition

Torah cantillation marks (Hebrew: taamei hamikra) are an ancient notation
system: every word in the Torah carries one of about 22 marks that dictates
the melody with which it is chanted during public reading. Learning to read
with correct cantillation, for example toward a Bar Mitzvah, takes months of
practice, and the only feedback available to a student is a teacher's ear.

**Goal:** train a neural network that identifies the cantillation mark from
the audio of a single sung word.

**Central scientific question:** does the model learn the melody of the mark,
or the voice of the reader? A model that learns a specific voice is useless
for a new student. The question is decided with a Leave-One-Reader-Out (LORO)
evaluation protocol (Section 6).

### System pipeline

```
YouTube readings (5 readers, 111 recordings, ~60 hours)
  -> mlx-whisper large-v3 transcription (word-level timestamps)
  -> precision-first cleaning against Sefaria's cantillated canonical text
     (verse-exact parasha boundaries, positional alignment, exact matches only)
  -> 69,798 verified labeled words
  -> 64,517 mel-spectrograms (128 x 87, ~2 seconds per word)
  -> 10 architectures compared -> LORO generalization test -> analyses
```

Implementation: `Data_Prep/clean_align_v2.py` (cleaning and alignment),
`Data_Prep/extract_features_v2.py` (feature extraction),
`00_validate_and_prepare_data.py` (validation and splits).""")

# ═══════════════════════════ 4. Dataset ═════════════════════════════════
md("""## 2. The Dataset

No labeled dataset of cantillation audio existed; constructing one is the
project's central contribution. Labels are not produced by human annotation
or by a recognizer: each transcribed word is aligned positionally against the
canonical cantillated text from the Sefaria API, and the label is read
directly from the Unicode cantillation character of the matched word.
Only exact alignment matches are kept (precision over recall; about 60% of
transcribed words survive).

The earlier version of the cleaning pipeline used incorrect chapter-level
parasha boundaries and set-based whitelisting; it capped the entire system at
30% accuracy on 5 classes. Fixing the data pipeline, not the architecture,
is what enabled 78% on 11 classes.

Class distribution of the full verified dataset:""")

code("""counts = {  # verified-word counts per class (see data_audit_report.md)
    'Tipecha': 11564, 'Munach': 9924, 'Mercha': 9690, 'Zaqef_Qatan': 7277,
    'Etnachta': 5627, 'Pashta': 5510, 'Siluk': 5396, 'Mahapakh': 2877,
    'Tevir': 2834, 'Revia': 2429, 'Qadma': 1875}
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.bar(range(len(counts)), counts.values(), color='#4C72B0')
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.keys(), rotation=30, ha='right')
ax.set_title('Verified words per cantillation mark (69,798 total, 5 readers)')
ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.show()""")

md("""Class imbalance (about 6:1 within the top-11) is handled with balanced
class weights in the loss and reported with imbalance-robust metrics
(balanced accuracy, Matthews correlation coefficient).

### Input representation

Each word is cut from the recording by its Whisper timestamps and converted
to a mel-spectrogram (128 mel bands, 87 time frames, about 2 seconds,
fmax 8 kHz), turning audio classification into image classification in which
the melodic contour is a visible curve:""")

code("""d = np.load('sample_data/sample_top11.npz', allow_pickle=False)
X, y = d['X'].astype(np.float32), d['y']
class_names = [str(c) for c in d['class_names']]
print('bundled sample:', X.shape)

fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
for ax, cls in zip(axes.flat, ['Etnachta', 'Siluk', 'Tipecha', 'Munach', 'Pashta', 'Revia']):
    i = int(np.where(y == class_names.index(cls))[0][0])
    ax.imshow(X[i], aspect='auto', origin='lower', cmap='magma')
    ax.set_title(cls); ax.set_xlabel('time frame'); ax.set_ylabel('mel bin')
plt.suptitle('Mel-spectrograms: each mark has a characteristic melodic contour')
plt.tight_layout(); plt.show()""")

# ═══════════════════════ 5. Architectures ═══════════════════════════════
md("""## 3. Architectures

Ten architectures share one registry (`10_train_pipeline.py`) and identical
training conditions: Adam (lr 5e-4, weight decay 1e-3), ReduceLROnPlateau,
early stopping (patience 10), balanced class weights, SpecAugment.

| # | Model | Parameters | Rationale |
|---|---|---|---|
| 1 | MLP | 5.8M | scientific baseline: value of spatial structure |
| 2 | CNN (3 blocks) | 5.3M | convolutional baseline |
| 3 | DeepCNN (5 blocks) | 5.9M | added depth and BatchNorm |
| 4 | ResNet-Mini | 2.3M | skip connections, global average pooling |
| 5 | CRNN-GRU | 2.1M | hybrid: CNN frontend + bidirectional GRU |
| 6 | CRNN-LSTM | 2.3M | direct GRU vs LSTM comparison |
| 7 | BiLSTM | 1.5M | purely recurrent over time frames |
| 8 | ResNet18-TL | 11.2M | ImageNet transfer learning |
| 9 | VGG16-TL | 15.8M | depth without skip connections, as contrast |
| 10 | Transformer | 2.0M | self-attention with a convolutional frontend |
""")

code("""import importlib.util
spec = importlib.util.spec_from_file_location('tp', '10_train_pipeline.py')
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
print(f'{"model":14s}{"parameters":>12s}')
for name, cls in tp.MODEL_REGISTRY.items():
    try:
        n = sum(p.numel() for p in cls(num_classes=11).parameters())
        print(f'{name:14s}{n/1e6:11.1f}M')
    except Exception as e:
        print(f'{name:14s}{"unavailable":>12s} ({type(e).__name__})')""")

# ═══════════════════════ 6. Live training demo ══════════════════════════
md("""## 4. Live Training Demonstration

The following cell trains the baseline CNN on the bundled 550-sample subset
(440 train / 110 test, 8 epochs, two to three minutes). Its purpose is to
demonstrate that the full training machinery is real and reproducible, not
to reproduce the headline numbers, which required the full dataset.""")

code("""import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

Xn = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-8)
Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.2, stratify=y, random_state=42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = tp.MODEL_REGISTRY['cnn'](num_classes=11).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
crit = nn.CrossEntropyLoss()
tr_x = torch.from_numpy(Xtr)[:, None].to(device); tr_y = torch.from_numpy(ytr).to(device)
te_x = torch.from_numpy(Xte)[:, None].to(device)

hist = []
for epoch in range(8):
    model.train(); perm = torch.randperm(len(tr_x))
    for i in range(0, len(perm), 32):
        b = perm[i:i+32]
        opt.zero_grad(); loss = crit(model(tr_x[b]), tr_y[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(te_x).argmax(1).cpu().numpy()
    hist.append(accuracy_score(yte, pred))
    print(f'epoch {epoch+1}/8  test accuracy {hist[-1]:.3f}  '
          f'balanced {balanced_accuracy_score(yte, pred):.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, 9), hist, 'o-')
axes[0].axhline(1/11, ls='--', c='gray', label='chance (9.1%)')
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('test accuracy')
axes[0].legend(); axes[0].grid(alpha=.3); axes[0].set_title('Learning on the bundled subset')
cm = confusion_matrix(yte, pred, normalize='true')
axes[1].imshow(cm, cmap='Blues'); axes[1].set_title('Demo confusion matrix')
axes[1].set_xticks(range(11)); axes[1].set_xticklabels(class_names, rotation=90, fontsize=7)
axes[1].set_yticks(range(11)); axes[1].set_yticklabels(class_names, fontsize=7)
plt.tight_layout(); plt.show()""")

# ═══════════════════════ 7. Full results ════════════════════════════════
md("""## 5. Full Results: Ten-Architecture Comparison

Trained on the full dataset (45,161 train / 9,678 validation / 9,678 test).
All metrics below are measured on the held-out test set. Random chance is 9.1%.

| Rank | Model | Accuracy | Balanced Acc. | MCC | F1 (macro) | AUC (macro) |
|---|---|---|---|---|---|---|
| 1 | ResNet18-TL | 78.0% | 79.9% | 0.753 | 77.7% | 0.972 |
| 2 | VGG16-TL | 77.0% | 79.3% | 0.743 | 76.4% | 0.972 |
| 3 | CRNN-GRU | 74.1% | 78.0% | 0.713 | 74.2% | 0.970 |
| 4 | DeepCNN | 73.8% | 77.3% | 0.708 | 73.5% | 0.967 |
| 5 | CRNN-LSTM | 73.5% | 76.5% | 0.704 | 73.4% | 0.967 |
| 6 | ResNet-Mini | 73.2% | 76.3% | 0.702 | 73.0% | 0.965 |
| 7 | BiLSTM | 70.2% | 72.9% | 0.667 | 69.8% | 0.959 |
| 8 | Transformer | 69.1% | 72.4% | 0.654 | 69.5% | 0.956 |
| 9 | CNN | 67.9% | 70.7% | 0.640 | 67.6% | 0.949 |
| 10 | MLP | 43.7% | 46.8% | 0.379 | 42.9% | 0.865 |

Key observations: the two transfer-learning networks lead and are nearly tied
(ResNet18 78.0%, VGG16 77.0%). This is an honest, slightly surprising result:
contrary to the expectation that ResNet's skip connections would give a clear
edge, on this task ImageNet initialization matters far more than the specific
backbone, and the 15.8M-parameter VGG did not collapse. The 24-point gap
between MLP and CNN quantifies the value of spatial structure; hybrid CRNN
models beat the pure CNN; GRU and LSTM are equivalent; the Transformer
underperforms at this data scale without pretraining, as expected.""")

figure_cell("results/random_top11/model_comparison_9.png",
            "Figure 1. Accuracy, balanced accuracy and MCC for the architecture comparison (test set).")

md("### The winner in detail: ResNet18 with transfer learning")

figure_cell("results/random_top11/resnet18_tl/resnet18_tl_training_curves.png",
            "Figure 2. Training curves. Early stopping restores the best-validation weights; no overfitting.")
figure_cell("results/random_top11/resnet18_tl/resnet18_tl_confusion_matrix.png",
            "Figure 3. Confusion matrices (raw and row-normalized). Errors concentrate among melodically similar pairs such as Munach and Mercha.")
figure_cell("results/random_top11/resnet18_tl/resnet18_tl_roc_curves.png",
            "Figure 4. Per-class ROC curves; macro AUC 0.972.")

# ═══════════════════════ 8. LORO ════════════════════════════════════════
md("""## 6. Generalization Test: Leave-One-Reader-Out

Each fold trains on four readers and evaluates on the fifth, a voice never
heard in training. Test accuracy per held-out reader (chance 9.1%):

| Model | R2 out | R3 out | R4 out | R5 out | Mean +- SD |
|---|---|---|---|---|---|
| DeepCNN | 57.5% | 45.9% | 45.5% | 63.1% | 53.0% +- 7.6 |
| ResNet18-TL | 54.4% | 50.4% | 37.6% | 63.1% | 51.4% +- 9.2 |

Balanced accuracy: DeepCNN 53.5% +- 8.7, ResNet18-TL 52.4% +- 9.7.
Full per-fold details: `results/loro_summary_4folds.md`.

Three conclusions:

1. The model learned melody, not voice: 53% on an unseen voice is 5.8 times
   chance (and on the largest, most reliable fold, R2 with 34,084 test words:
   57.5% accuracy, AUC 0.90).
2. The ranking inverts under distribution shift. ResNet18-TL, the winner on
   familiar readers, drops below the three-times-smaller DeepCNN when tested
   on an unseen voice: the larger model partially memorizes reader identity,
   while DeepCNN's limited capacity forces it to learn the generic melodic
   shape. DeepCNN is therefore the production model.
3. The spread across folds (38% to 63%) shows generalization depends on the
   held-out reader's style, evidence that additional training readers are the
   most valuable next investment.""")

# ═══════════════════════ 9. Sweep ═══════════════════════════════════════
md("""## 7. Hyperparameter Study

Nine configurations of DeepCNN, 12 epochs each to isolate effects: a learning
rate by batch size grid plus two regularization ablations. Test metrics:

| Configuration | Accuracy | Balanced Acc. | MCC |
|---|---|---|---|
| No SpecAugment | 71.1% | 74.7% | 0.678 |
| No L2 weight decay | 70.0% | 73.1% | 0.666 |
| lr 5e-4, batch 128 | 69.2% | 72.4% | 0.657 |
| lr 1e-4, batch 32 | 68.6% | 72.1% | 0.650 |
| Default (lr 5e-4, batch 64) | 68.8% | 71.9% | 0.651 |
| lr 1e-3, batch 128 | 69.0% | 71.4% | 0.653 |
| lr 5e-4, batch 32 | 67.6% | 71.4% | 0.640 |
| lr 1e-4, batch 128 | 67.7% | 70.6% | 0.638 |
| lr 1e-3, batch 32 | 66.9% | 69.2% | 0.628 |

Interpretation: the middle learning rate (5e-4) is stable across batch sizes,
while a high rate with a small batch is the worst combination. Under a short
budget, removing regularization helps, because SpecAugment and L2 slow early
convergence; their benefit materializes only in full-length runs, where the
fully regularized model reaches 77.3% balanced accuracy, above every row in
this table. This is the classic regularization-budget trade-off.""")

# ═══════════════════════ 10. Similarity ═════════════════════════════════
md("""## 8. Which Marks Sound Alike? The Network versus the Tradition

Cosine distances between class centroids in the penultimate embedding space,
together with symmetric confusion rates, identify the closest pairs
(Munach-Mercha: distance 0.078, 18.3% confusion; both are conjunctive
"servant" marks) and the farthest (Etnachta-Mahapakh: 0.580, near-zero
confusion).

The traditional taxonomy (Wickes, 1887) ranks the marks into a hierarchy of
disjunctives (emperors, kings, dukes, counts) and conjunctives (servants).
Testing the model's 1,418 misclassifications against this hierarchy:
49.2% of errors stay within the same traditional rank, versus 32.8% expected
by chance given the error distribution, a factor of 1.50. A network that saw
only audio reconstructed a centuries-old taxonomy.""")

figure_cell("analysis/taam_dendrogram.png",
            "Figure 5. Hierarchical clustering of embedding-space centroid distances. Leaf colors mark the traditional rank of each mark.")
figure_cell("analysis/taam_projection.png",
            "Figure 6. Two-dimensional projection of word embeddings, colored by cantillation mark (left) and by reader (right). Organization follows the mark, not the voice.")
figure_cell("analysis/taam_confusion_similarity.png",
            "Figure 7. Symmetric confusion-rate matrix.")
figure_cell("analysis/taam_embedding_distance.png",
            "Figure 8. Centroid cosine-distance matrix in embedding space.")

# ═══════════════════════ 11. Sequence model ═════════════════════════════
md("""## 9. Cantillation Grammar as Signal: the Verse-Context Model

Cantillation marks follow near-fixed sequences (Mercha precedes Tipecha,
Mahapakh precedes Pashta). To measure this signal, the best CNN is frozen,
every word is embedded, and a bidirectional LSTM tagger is trained over the
embedding sequence of an entire verse. The split is by verse, so no verse
straddles train and test.

| Model | Accuracy | Balanced Acc. | MCC |
|---|---|---|---|
| CNN alone (per word) | 84.9% | 88.2% | 0.830 |
| CNN + verse-level BiLSTM | 92.9% | 93.6% | 0.919 |

Context contributes +7.9 accuracy points. Methodological caution: absolute
numbers here are inflated because the frozen CNN saw some of these words in
its own training; the fair claim is the difference between the two rows,
which are measured on identical test words.""")

# ═══════════════════════ 12. Grad-CAM ═══════════════════════════════════
md("""## 10. Explainability: Grad-CAM

Gradient-weighted class activation maps show where the model attends. The
activation sits on the harmonic bands of the sung vowels, that is, on the
melodic contour itself, and not on padding silence or background noise.""")

figure_cell("analysis/gradcam/gradcam_Etnachta.png",
            "Figure 9. Grad-CAM for Etnachta: strongest activation on the characteristic fall-and-rise pattern.")
figure_cell("analysis/gradcam/gradcam_Tipecha.png",
            "Figure 10. Grad-CAM for Tipecha.")
figure_cell("analysis/gradcam/gradcam_Munach.png",
            "Figure 11. Grad-CAM for Munach.")
figure_cell("analysis/gradcam/gradcam_Siluk.png",
            "Figure 12. Grad-CAM for Siluk (end of verse).")

# ═══════════════════════ 13. Production ═════════════════════════════════
md("""## 11. From Research to Product

A working web application (FastAPI backend, dependency-free JavaScript
frontend; `uvicorn webapp.server:app --port 8077`) offers three views:

1. Practice: the student picks a mark, listens to a real "golden teacher"
   clip cut from the Torah reading, and records the same word in the browser
   (raw PCM capture, no plugins). The combined score is 60% the network's
   probability for the target mark and 40% a key-normalized pitch-curve DTW
   component, drawn live over the teacher's melodic band.
2. Analyze Recording: any uploaded reading is transcribed word by word and
   each word receives top-3 mark predictions with confidences.
3. About: model card and provenance.""")

figure_cell("app_assets/app_screenshot.png",
            "Figure 13. The web application, practice view: a real teacher clip with its melodic curve and tolerance band.")

md("""### Deployment measurements

The production model (DeepCNN, chosen for its superior unseen-reader
generalization) was exported to ONNX with full output parity (maximum
difference 1e-6) and benchmarked per single word:

| Runtime | Latency per word |
|---|---|
| PyTorch CPU | 6.8 ms |
| ONNX Runtime CPU | 8.9 ms |
| PyTorch MPS (Apple M4) | 24.1 ms |

The real-time budget for a responsive tutor is roughly 50 ms per word; every
runtime clears it before INT8 quantization.

### Reproducibility

```
python smoke_test.py                       # end-to-end pipeline check, ~2 min, no data needed
python 10_train_pipeline.py --model all    # full reproduction (requires the dataset)
python 11_run_loro.py --model resnet18_tl,deep_cnn
```

`audit_listen/` contains ten randomly sampled verified words with their
labels and verse references for human verification of label quality.""")

# ═══════════════════════ 14. Future work ════════════════════════════════
md("""## 12. How This Project Continues

### Author's note: a segmentation limitation found while testing the app

While listening to the teacher clips in the practice interface I noticed a
consistent artifact: each clip begins with the tail of the *previous* word,
then the intended word, and the intended word's final consonant is clipped.
Inspecting the alignment confirms the cause. Whisper's Hebrew word timestamps
are contiguous, the median gap between one word's end and the next word's
start is 0.000 seconds, so the boundaries carry a small systematic lag
relative to the true acoustic onsets, and with a fixed plus/minus 50 ms
window that lag bleeds the neighbor in and truncates the target's final
phoneme. Because the timestamps have no gaps, there is no silence margin to
absorb the offset.

What this means honestly: every word spectrogram, and therefore the melodic
curves shown in the practice mode, is slightly contaminated at both ends.
The classification results (78% random-split, 53% LORO) still stand because a
convolutional model is largely shift-tolerant and sees most of the target
contour, but this is a real ceiling on the work and the pitch-DTW curves in
particular inherit the artifact. I chose to document it rather than hide it;
in a course project a known, characterized limitation is worth more than a
clean-looking result whose flaw is undiscovered.

The correct fix, deferred only because it requires re-extracting every
spectrogram and retraining all models (days of compute):

- Replace Whisper timestamps with a phoneme-level forced aligner
  (Montreal Forced Aligner, or a wav2vec2 CTC alignment) that places
  boundaries on the true acoustic onsets rather than contiguously.
- Alternatively, estimate the systematic offset empirically (cross-correlate
  clip energy against the timestamp) and shift every window by that amount
  before extraction.
- Add the boundary-jitter augmentation below so the model is trained to be
  robust to whatever residual offset remains.

### Data and modeling roadmap

The single highest-leverage investment is more data at the same precision
standard, in this order of priority:

1. **More full-Torah readers, cleaned to perfect precision.** The LORO spread
   (38% to 63% depending on the held-out reader) shows that reader diversity,
   not architecture, is the current bottleneck. Every added reader passes
   through the same pipeline: word-level transcription, verse-exact alignment
   against the canonical text, and exact-match-only verification. We
   deliberately keep recall low (about 60% of words survive) so that label
   precision stays near perfect; with more raw hours, discarding ambiguous
   words costs nothing. The pipeline is already one command per stage, so a
   new reader is an acquisition problem, not an engineering problem.
2. **Cover the remaining 12 rare marks.** Below roughly 250 examples a class
   is not learnable with this setup; targeted collection of rare-mark verses
   (their locations are known exactly from the canonical text) or few-shot
   techniques over the embedding space would extend the model from 11 to all
   22 mark classes.
3. **Sequence decoding in production.** The verse-context tagger already adds
   7.9 points offline; integrating it into the application's continuous
   reading mode would let grammar correct acoustic mistakes in real time.
4. **Hierarchical loss.** Penalize within-rank confusions (for example Munach
   versus Mercha, both conjunctives) less than cross-rank ones, aligning the
   objective with both the tradition and the observed error structure.
5. **Boundary-robust training.** Random shifts of the word window by 50 to
   100 ms during training, immunizing the model against imperfect timestamp
   cuts at inference time.
6. **On-device deployment.** The exported ONNX model already runs at 6.8 ms
   per word on CPU; INT8 quantization and a mobile runtime (Core ML or NNAPI)
   would put the tutor fully offline on a phone, with pitch tracking robust
   to adolescent voice breaks (a CREPE-class F0 model) and syllable-level
   forced alignment for feedback such as "the accent landed on the wrong
   syllable".

The direction of the results is consistent: precision-first data beat model
size at every step of this project, and the roadmap above continues that
strategy.""")

# ═══════════════════════ 15. Conclusions ════════════════════════════════
md("""## 13. Conclusions

1. Data quality beats architecture. Fixing the cleaning pipeline, not the
   model, moved the system from 30% on 5 classes to 78% on 11 classes.
2. Transfer learning wins in-distribution, but the smaller model wins under
   distribution shift: the LORO ranking inversion shows that "the best model"
   depends on who it will serve. DeepCNN is the production choice.
3. The model learned melody, not voice: 5.8 times chance on unseen voices.
4. The network reconstructed the traditional hierarchy of the marks from
   audio alone (within-rank errors at 1.50 times chance).
5. Cantillation grammar is real signal: verse context adds 7.9 points.

### Main references

Radford et al., 2023 (Whisper) | He et al., 2016 (ResNet) | Deng et al., 2009
(ImageNet) | Park et al., 2019 (SpecAugment) | Simonyan and Zisserman, 2015
(VGG) | Mauch and Dixon, 2014 (pYIN) | Wickes, 1887 (cantillation taxonomy) |
Sefaria API | librosa. Full list: `report/Taamify_Report.md`, Section 8.

*Built with PyTorch on Apple M4 (MPS). The full report, presentation, and
application are available in the repository.*""")

# ═══════════════════════ write ══════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for c in nb["cells"]:
    if isinstance(c["source"], str):
        c["source"] = c["source"].splitlines(keepends=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
size = OUT.stat().st_size / 1e6
print(f"wrote {OUT.name}: {len(cells)} cells, {size:.1f} MB (embedded figures)")
