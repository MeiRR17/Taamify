"""
analysis_taam_similarity.py — Which ta'amim sound alike?
========================================================
Answers two questions from a trained model:
1. Which ta'amim are melodically closest / farthest apart?
   (penultimate-embedding centroid distances + hierarchical dendrogram,
    plus symmetric confusion rates)
2. Does the embedding space organize by TA'AM or by READER?
   (2D projection colored both ways — the generalization story in one figure)

Runs the model over the per-reader feature files (features_v2/), so reader
identity is available — unlike the mixed test-set embeddings export.

Usage:
  python analysis_taam_similarity.py --checkpoint results/random_top11/cnn/cnn_best.pth
  python analysis_taam_similarity.py --checkpoint ... --readers R1 R2 --max-per-class 400
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_DIR = Path(__file__).parent
# TAAMIFY_FEATURES lets long runs read a local copy instead of the
# Google-Drive-synced folder (Drive eviction corrupts long reads)
FEATURES_DIR = Path(os.environ.get("TAAMIFY_FEATURES",
                                   PROJECT_DIR / "features_v2"))
ANALYSIS_DIR = PROJECT_DIR / "analysis"

# Fixed categorical order/colors per class index (never re-cycled between plots)
CLASS_CMAP = plt.cm.tab20

# Traditional disjunctive hierarchy of the ta'amim (Wickes' taxonomy):
# emperors > kings > dukes > counts stop the reading with decreasing force;
# servants (conjunctives) connect a word to the next. The scientific question:
# does a model trained only on AUDIO rediscover this centuries-old taxonomy?
TRADITIONAL_RANKS = {
    'Siluk': 'Emperor', 'Etnachta': 'Emperor',
    'Segol': 'King', 'Shalshelet': 'King', 'Zaqef_Qatan': 'King',
    'Zaqef_Gadol': 'King', 'Tipecha': 'King', 'Revia': 'King',
    'Zarqa': 'Duke', 'Pashta': 'Duke', 'Yetiv': 'Duke', 'Tevir': 'Duke',
    'Geresh': 'Count', 'Gershayim': 'Count', 'Pazer': 'Count',
    'Telisha_Gedola': 'Count',
    'Munach': 'Servant', 'Mahapakh': 'Servant', 'Mercha': 'Servant',
    'Darga': 'Servant', 'Qadma': 'Servant', 'Telisha_Qetana': 'Servant',
    'Mercha_Kefula': 'Servant',
}
RANK_ORDER = ['Emperor', 'King', 'Duke', 'Count', 'Servant']
RANK_HEBREW = {'Emperor': 'קיסרים', 'King': 'מלכים', 'Duke': 'משנים',
               'Count': 'שלישים', 'Servant': 'משרתים'}
RANK_COLORS = {'Emperor': '#B2182B', 'King': '#EF8A62', 'Duke': '#67A9CF',
               'Count': '#2166AC', 'Servant': '#5AAE61'}


def rank_of(name):
    return TRADITIONAL_RANKS.get(name, 'Servant')


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_pipeline", PROJECT_DIR / "10_train_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(checkpoint_path: Path, train_mod):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch = ckpt['architecture']
    class_names = ckpt['class_names']
    model = train_mod.MODEL_REGISTRY[arch](num_classes=len(class_names))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, arch, class_names


def collect_embeddings(model, class_names, readers, device, max_per_class):
    """Run per-reader features through the model; return embeddings, labels,
    reader ids, and predictions."""
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    rng = np.random.default_rng(42)

    X_parts, y_parts, r_parts = [], [], []
    for reader in readers:
        x_path = FEATURES_DIR / f"{reader}_X.npy"
        m_path = FEATURES_DIR / f"{reader}_meta.json"
        if not x_path.exists():
            print(f"  {reader}: no features, skipping")
            continue
        X = np.load(x_path).astype(np.float32)
        with open(m_path, encoding='utf-8') as f:
            meta = json.load(f)
        taams = np.array([m['taam'] for m in meta])
        mask = np.isin(taams, class_names)
        X, taams = X[mask], taams[mask]
        # cap per class per reader to keep the projection legible
        keep = []
        for cls in class_names:
            idx = np.where(taams == cls)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            keep.extend(idx)
        keep = np.array(sorted(keep))
        X, taams = X[keep], taams[keep]
        X_parts.append(X)
        y_parts.append(np.array([label_to_idx[t] for t in taams]))
        r_parts.append(np.array([reader] * len(X)))
        print(f"  {reader}: {len(X)} samples")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    readers_arr = np.concatenate(r_parts)

    # normalize per sample — must match 00_validate_and_prepare_data
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    std[std == 0] = 1.0
    X = ((X - mean) / std)[:, None, :, :]

    # forward pass with penultimate hook
    last_linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    captured, logits_all = [], []

    def hook(_m, inputs, output):
        captured.append(inputs[0].detach().cpu())
        logits_all.append(output.detach().cpu())

    handle = last_linear.register_forward_hook(hook)
    model.to(device)
    with torch.no_grad():
        for i in range(0, len(X), 256):
            model(torch.from_numpy(X[i:i + 256]).to(device))
    handle.remove()

    emb = torch.cat(captured).numpy()
    preds = torch.cat(logits_all).argmax(dim=1).numpy()
    return emb, y, readers_arr, preds


def centroid_distance_matrix(emb, y, num_classes):
    """Cosine distance between class centroids."""
    centroids = np.stack([emb[y == i].mean(axis=0) for i in range(num_classes)])
    normed = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return 1.0 - normed @ normed.T


def symmetric_confusion(y_true, y_pred, num_classes):
    """P(i confused with j), symmetrized."""
    cm = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    rate = cm / row
    return (rate + rate.T) / 2


def plot_heatmap(matrix, class_names, title, cbar_label, path, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': cbar_label},
                annot_kws={'fontsize': 8})
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_dendrogram(dist, class_names, path):
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    fig, ax = plt.subplots(figsize=(11, 6.5))
    dendrogram(Z, labels=class_names, ax=ax, leaf_rotation=45,
               color_threshold=0.7 * condensed.max())
    # color each leaf label by its traditional rank — lets the eye check
    # whether acoustic clustering matches the centuries-old taxonomy
    for lbl in ax.get_xticklabels():
        lbl.set_color(RANK_COLORS[rank_of(lbl.get_text())])
        lbl.set_fontweight('bold')
    handles = [plt.Line2D([0], [0], marker='s', linestyle='', markersize=9,
                          color=RANK_COLORS[r],
                          label=f"{r} ({RANK_HEBREW[r]})") for r in RANK_ORDER
               if any(rank_of(c) == r for c in class_names)]
    ax.legend(handles=handles, title='Traditional rank', loc='upper right',
              fontsize=9, framealpha=0.9)
    ax.set_title("Melodic similarity of ta'amim — hierarchical clustering\n"
                 "(cosine distance between embedding centroids; label color = traditional rank)",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Cosine distance')
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_projection(emb, y, readers_arr, class_names, path):
    """2D projection, colored by ta'am (left) and by reader (right)."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1)
        coords = reducer.fit_transform(emb)
        method = 'UMAP'
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42,
                      perplexity=30).fit_transform(emb)
        method = 't-SNE'

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    ax = axes[0]
    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.55,
                   color=CLASS_CMAP(i % 20), label=name, linewidths=0)
    ax.set_title(f'{method} of embeddings — colored by TA\'AM', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=9, loc='best', framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ranks = np.array([rank_of(class_names[i]) for i in y])
    for r in RANK_ORDER:
        mask = ranks == r
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.55,
                   color=RANK_COLORS[r], label=f"{r} ({RANK_HEBREW[r]})", linewidths=0)
    ax.set_title(f'{method} — colored by TRADITIONAL RANK', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    reader_names = sorted(set(readers_arr))
    for i, r in enumerate(reader_names):
        mask = readers_arr == r
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.55,
                   color=plt.cm.Set2(i % 8), label=r, linewidths=0)
    ax.set_title(f'{method} of embeddings — colored by READER', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle('If the left panel shows structure and the right panel is mixed,\n'
                 'the model learned the MELODY, not the VOICE',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def rank_error_analysis(y_true, preds, class_names):
    """Of all misclassifications, what share stays within the same traditional
    rank? Compared against a chance baseline that respects how often each class
    is (wrongly) predicted overall."""
    from collections import Counter
    errors = [(t, p) for t, p in zip(y_true, preds) if t != p]
    if not errors:
        return None
    n = len(class_names)
    within = sum(1 for t, p in errors
                 if rank_of(class_names[t]) == rank_of(class_names[p]))
    observed = within / len(errors)

    pred_counts = Counter(p for _, p in errors)
    total = sum(pred_counts.values())
    q = {c: pred_counts.get(c, 0) / total for c in range(n)}
    chance = 0.0
    true_counts = Counter(t for t, _ in errors)
    for t, n_t in true_counts.items():
        denom = sum(q[j] for j in range(n) if j != t)
        same = sum(q[j] for j in range(n) if j != t
                   and rank_of(class_names[j]) == rank_of(class_names[t]))
        chance += (n_t / len(errors)) * (same / denom if denom else 0.0)
    return observed, chance, len(errors)


def write_report(dist, conf_sim, class_names, arch, readers, path, rank_stats=None):
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((dist[i, j], conf_sim[i, j], class_names[i], class_names[j]))
    by_emb = sorted(pairs)

    lines = [f"# Ta'am Similarity Report", "",
             f"Model: {arch} | Readers: {', '.join(readers)}", "",
             "Embedding distance = cosine distance between class centroids in the "
             "model's penultimate layer (0 = identical melody as the model hears it).",
             "Confusion rate = symmetric share of samples the model mixes up between the pair.",
             "", "## Closest pairs (hardest to tell apart)", "",
             "| Pair | Embedding distance | Confusion rate |", "|---|---|---|"]
    for d, c, a, b in by_emb[:8]:
        lines.append(f"| {a} — {b} | {d:.3f} | {c:.1%} |")
    lines += ["", "## Farthest pairs (most distinct melodies)", "",
              "| Pair | Embedding distance | Confusion rate |", "|---|---|---|"]
    for d, c, a, b in by_emb[-8:][::-1]:
        lines.append(f"| {a} — {b} | {d:.3f} | {c:.1%} |")

    if rank_stats is not None:
        observed, chance, n_err = rank_stats
        lines += ["", "## Does the model rediscover the traditional taxonomy?", "",
                  f"Ranks: Emperors (קיסרים) > Kings (מלכים) > Dukes (משנים) > "
                  f"Counts (שלישים); Servants (משרתים) are conjunctives.", "",
                  f"- Misclassifications analyzed: {n_err}",
                  f"- **{observed:.1%} of errors stay within the same traditional rank**",
                  f"- Chance baseline (given overall error distribution): {chance:.1%}",
                  f"- Ratio: **{observed / chance:.2f}× above chance** — "
                  + ("the model's acoustic confusions align with the centuries-old "
                     "hierarchy" if observed > chance else
                     "no alignment with the traditional hierarchy")]

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report -> {path}")
    print("\n".join(lines[:20]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True,
                    help='trained model .pth from 10_train_pipeline.py')
    ap.add_argument('--readers', nargs='+', default=['R1', 'R2', 'R3', 'R4', 'R5'])
    ap.add_argument('--max-per-class', type=int, default=300,
                    help='per reader, to keep the projection legible')
    args = ap.parse_args()

    train_mod = load_train_module()
    device = train_mod.pick_device()
    model, arch, class_names = load_model(args.checkpoint, train_mod)
    print(f"Model: {arch} | classes: {class_names} | device: {device}")

    print("\nCollecting embeddings:")
    emb, y, readers_arr, preds = collect_embeddings(
        model, class_names, args.readers, device, args.max_per_class)
    print(f"Total: {emb.shape[0]} samples, dim {emb.shape[1]}")

    ANALYSIS_DIR.mkdir(exist_ok=True)
    n = len(class_names)

    dist = centroid_distance_matrix(emb, y, n)
    conf_sim = symmetric_confusion(y, preds, n)

    plot_heatmap(dist, class_names,
                 "Embedding centroid distance (cosine) — lower = more similar melody",
                 'Cosine distance', ANALYSIS_DIR / 'taam_embedding_distance.png')
    plot_heatmap(conf_sim, class_names,
                 "Symmetric confusion rate — how often the model mixes the pair up",
                 'Confusion rate', ANALYSIS_DIR / 'taam_confusion_similarity.png')
    plot_dendrogram(dist, class_names, ANALYSIS_DIR / 'taam_dendrogram.png')
    plot_projection(emb, y, readers_arr, class_names, ANALYSIS_DIR / 'taam_projection.png')
    rank_stats = rank_error_analysis(y, preds, class_names)
    write_report(dist, conf_sim, class_names, arch, args.readers,
                 ANALYSIS_DIR / 'taam_similarity_report.md', rank_stats)

    print(f"\nAll analysis outputs -> {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
