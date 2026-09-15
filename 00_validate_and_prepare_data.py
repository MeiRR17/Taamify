"""
00_validate_and_prepare_data.py  (v2 — multi-reader, LORO-aware)
================================================================
Validates and packages the v2 features (features_v2/<reader>_X.npy + _meta.json,
produced by Data_Prep/extract_features_v2.py) into ready-to-train splits.

1. Loads all requested readers, selects top-N ta'am classes by combined count
2. Validates every spectrogram (shape, NaN/Inf, energy, padding ratio, variance)
3. Normalizes per sample (zero-mean, unit-variance)
4. Splits:
     --split random          stratified 70/15/15 (default)
     --split loro --holdout R2
                             held-out reader = test set; the rest -> 85/15 train/val
5. Saves prepared_data/<scenario>/{train,val,test}.npz + metadata.json + quality report

Usage:
    python 00_validate_and_prepare_data.py --num-classes 11
    python 00_validate_and_prepare_data.py --num-classes 11 --split loro --holdout R2
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
# TAAMIFY_FEATURES lets long runs read a local copy instead of the
# Google-Drive-synced folder (Drive eviction corrupts long reads)
FEATURES_DIR = Path(os.environ.get("TAAMIFY_FEATURES",
                                   PROJECT_DIR / "features_v2"))
OUTPUT_ROOT = PROJECT_DIR / "prepared_data"

N_MELS = 128
MAX_FRAMES = 87
MIN_ENERGY_THRESHOLD = -75.0  # dB — mean below this is likely silence
MAX_PAD_RATIO = 0.85          # fraction of frames allowed to be pure padding
PAD_DB = -80.0                # silence floor used by extract_features_v2


def validate_sample(spec: np.ndarray) -> str | None:
    """Return rejection reason, or None if the spectrogram is valid."""
    if spec.shape != (N_MELS, MAX_FRAMES):
        return 'bad_shape'
    if np.isnan(spec).any() or np.isinf(spec).any():
        return 'contains_nan_or_inf'
    if spec.mean() < MIN_ENERGY_THRESHOLD:
        return 'too_silent'
    # padding = frames entirely at the dB floor (also catches v1-style all-zero frames)
    pad_frames = np.all(spec <= PAD_DB + 0.1, axis=0) | np.all(spec == 0, axis=0)
    if pad_frames.mean() > MAX_PAD_RATIO:
        return 'too_much_padding'
    if spec.std() < 0.01:
        return 'no_variance'
    return None


def load_readers(readers: list) -> tuple:
    """Load and concatenate all readers. Returns (X, taam_labels, reader_labels,
    align_methods) -- the last one is each word's 'align' meta field
    (e.g. 'verse_ctc' vs 'stored', the Whisper-boundary fallback that never
    got replaced by verse-level alignment), for FINDINGS.md §8's pending
    boundary-quality ablation: is the R1 wall partly a boundary-quality
    artifact, not purely a voice/generalization one?"""
    X_parts, taams, reader_ids, aligns, parashot = [], [], [], [], []
    for reader in readers:
        x_path = FEATURES_DIR / f"{reader}_X.npy"
        m_path = FEATURES_DIR / f"{reader}_meta.json"
        if not x_path.exists():
            print(f"  WARNING: {x_path.name} not found, skipping {reader}")
            continue
        X = np.load(x_path).astype(np.float32)
        with open(m_path, encoding='utf-8') as f:
            meta = json.load(f)
        assert len(meta) == len(X), f"{reader}: meta/X length mismatch"
        X_parts.append(X)
        taams.extend(m['taam'] for m in meta)
        reader_ids.extend([reader] * len(X))
        aligns.extend(m.get('align', 'unknown') for m in meta)
        parashot.extend(str(m.get('parasha', '?')) for m in meta)
        print(f"  {reader}: {len(X)} samples")
    if not X_parts:
        raise SystemExit("No feature files found — run Data_Prep/extract_features_v2.py first")
    return (np.concatenate(X_parts), np.array(taams), np.array(reader_ids),
            np.array(aligns), np.array(parashot))


def select_classes(taams: np.ndarray, num_classes: int | None) -> list:
    counts = Counter(taams)
    ranked = [name for name, _ in counts.most_common()]
    selected = ranked if num_classes is None else ranked[:num_classes]
    print(f"\nClasses ({len(selected)} of {len(ranked)}):")
    for name in selected:
        print(f"  {name:20s} {counts[name]:6d}")
    return selected


def validate_and_filter(X, taams, reader_ids, class_names, parashot=None):
    """Keep only valid samples of selected classes."""
    print(f"\n{'='*60}\nVALIDATING {len(X)} SPECTROGRAMS\n{'='*60}")
    class_set = set(class_names)
    keep_idx, reasons = [], Counter()
    for i in range(len(X)):
        if taams[i] not in class_set:
            reasons['other_class'] += 1
            continue
        reason = validate_sample(X[i])
        if reason:
            reasons[reason] += 1
            continue
        keep_idx.append(i)
    print(f"Valid: {len(keep_idx)} / {len(X)}")
    if reasons:
        for reason, n in reasons.most_common():
            print(f"  rejected {reason}: {n}")
    keep_idx = np.array(keep_idx)
    if parashot is not None:
        return (X[keep_idx], taams[keep_idx], reader_ids[keep_idx],
                parashot[keep_idx], reasons)
    return X[keep_idx], taams[keep_idx], reader_ids[keep_idx], reasons


def normalize_per_sample(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def make_random_splits(X, y, test_size, val_size, seed=42):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=seed, stratify=y_tv)
    return X_train, y_train, X_val, y_val, X_test, y_test


def balance_by_reader(X, y, reader_ids, rng_seed=42):
    """Cap every reader to the size of the smallest, so no single voice
    dominates the training pool.

    --max-train stratifies by CLASS but not by READER, which quietly confounds
    any 'does adding reader Z help?' experiment: BEN13R alone is 53.5% of the
    available pool, so a 40,000-sample cap draws ~21,000 words from one voice
    and the model can improve at recognising that voice while getting worse at
    everyone else. Balancing separates 'this corpus is bad' from 'this corpus
    is too big a share of the mix', which are different findings with different
    fixes."""
    import numpy as _np
    rng = _np.random.default_rng(rng_seed)
    readers = _np.unique(reader_ids)
    smallest = min((reader_ids == r).sum() for r in readers)
    keep = []
    for r in readers:
        idx = _np.flatnonzero(reader_ids == r)
        keep.append(rng.choice(idx, size=smallest, replace=False))
    keep = _np.sort(_np.concatenate(keep))
    print(f"  balanced to {smallest:,} per reader "
          f"({len(readers)} readers, {len(keep):,} total)")
    return X[keep], y[keep], reader_ids[keep]



def make_parasha_splits(X, y, parashot, n_test, val_size=0.15, seed=42,
                        draw_seed=None):
    """Hold out whole parashot, not random words.

    Ben13 is ONE voice. A random word-level split therefore leaks badly: the
    word before and the word after the test word come from the same verse, the
    same recording session, the same microphone and room, and often the same
    breath. A model can score well on that without having learned anything
    transferable. Holding out whole parashot means the test words come from
    audio the model never touched, which is the honest question for a
    single-voice corpus -- and the closest proxy available until a second
    Sephardic-Jerusalem voice is recorded.
    """
    # draw_seed picks WHICH parashot are held out, separately from the
    # train/val shuffle. Without it every '--tag drawN' run held out the
    # identical 8 parashot and three 'independent' draws returned the same
    # number to four decimals -- a test that measured nothing.
    rng = np.random.default_rng(seed if draw_seed is None else draw_seed)
    uniq = np.unique(parashot)
    if len(uniq) <= n_test + 1:
        raise SystemExit(f"only {len(uniq)} parashot present; cannot hold out {n_test}")
    test_p = set(rng.choice(uniq, size=n_test, replace=False).tolist())
    test_mask = np.isin(parashot, list(test_p))
    X_test, y_test = X[test_mask], y[test_mask]
    X_rest, y_rest = X[~test_mask], y[~test_mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, random_state=seed, stratify=y_rest)
    print(f"\nParasha split: {n_test} of {len(uniq)} parashot held out "
          f"({sorted(test_p)})")
    print(f"  train {len(X_train):,}  val {len(X_val):,}  test {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loro_splits(X, y, reader_ids, holdout, val_size=0.15, seed=42,
                     max_train=None):
    test_mask = reader_ids == holdout
    if not test_mask.any():
        raise SystemExit(f"holdout reader {holdout} has no samples")
    X_test, y_test = X[test_mask], y[test_mask]
    X_rest, y_rest = X[~test_mask], y[~test_mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, random_state=seed, stratify=y_rest)

    # The folds are not comparable out of the box: holding out the largest
    # reader leaves far less training data than holding out a small one, so a
    # weak fold could be starved rather than out-of-domain. --max-train caps
    # every fold to the same training size, stratified so the class balance is
    # untouched, which separates the two explanations.
    if max_train and max_train < len(X_train):
        keep, _ = train_test_split(np.arange(len(X_train)), train_size=max_train,
                                   random_state=seed, stratify=y_train)
        X_train, y_train = X_train[keep], y_train[keep]
        print(f"  capped training set to {max_train:,} samples (stratified)")

    train_readers = sorted(set(reader_ids[~test_mask]))
    print(f"\nLORO: train/val on {train_readers}, test on held-out {holdout}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_report(splits, class_names, reasons, output_dir: Path, scenario: str):
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    x = np.arange(len(class_names))
    width = 0.25
    tr = [(y_train == i).sum() for i in range(len(class_names))]
    va = [(y_val == i).sum() for i in range(len(class_names))]
    te = [(y_test == i).sum() for i in range(len(class_names))]
    ax = axes[0]
    ax.bar(x - width, tr, width, label='Train', color='#2196F3')
    ax.bar(x, va, width, label='Val', color='#FF9800')
    ax.bar(x + width, te, width, label='Test', color='#4CAF50')
    ax.set_title(f'Class Distribution — {scenario}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    idx = np.random.default_rng(0).integers(0, len(X_train))
    im = ax.imshow(X_train[idx, 0], aspect='auto', origin='lower', cmap='magma')
    ax.set_title(f'Sample: {class_names[y_train[idx]]}', fontweight='bold')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Mel Bin')
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    totals = [tr[i] + va[i] + te[i] for i in range(len(class_names))]
    ratios = [c / max(totals) for c in totals]
    colors = ['#f44336' if r < 0.25 else '#FF9800' if r < 0.5 else '#4CAF50' for r in ratios]
    ax.barh(class_names, ratios, color=colors)
    ax.set_title('Class Balance Ratio', fontweight='bold')
    ax.set_xlim(0, 1.15)
    for i, (r, c) in enumerate(zip(ratios, totals)):
        ax.text(r + 0.02, i, str(c), va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'data_quality_report.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    with open(output_dir / 'data_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"DATA PREPARATION REPORT — {scenario}\n{'='*60}\n\n")
        if reasons:
            f.write("Rejections during validation:\n")
            for reason, n in reasons.most_common():
                f.write(f"  {reason}: {n}\n")
        f.write(f"\nSplit sizes: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}\n\n")
        f.write(f"{'Class':22s} {'Train':>7s} {'Val':>7s} {'Test':>7s}\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:22s} {tr[i]:7d} {va[i]:7d} {te[i]:7d}\n")
    print(f"  Report -> {output_dir / 'data_quality_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Taamify data validation & preparation v2")
    parser.add_argument('--readers', nargs='+', default=['R1', 'R2', 'R3', 'R4', 'R5'])
    parser.add_argument('--num-classes', type=int, default=11)
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--split', choices=['random', 'loro', 'parasha'], default='random')
    parser.add_argument('--parasha-seed', type=int, default=None,
                        help='which parashot to hold out; vary it to get genuinely '
                             'independent draws (see make_parasha_splits)')
    parser.add_argument('--test-parashot', type=int, default=8,
                        help="parasha split: how many whole parashot to hold out. "
                             "With a single-voice corpus like Ben13 a random word "
                             "split leaks: neighbouring words of the same verse, "
                             "recorded in the same session under identical "
                             "acoustics, land on both sides. Holding out whole "
                             "parashot is the honest test of whether the model "
                             "learned the ta'amim or the recordings.")
    parser.add_argument('--holdout', default=None, help='reader to hold out (loro)')
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--max-train', type=int, default=None,
                        help='cap the training set (loro), so folds with very '
                             'different training sizes can be compared')
    parser.add_argument('--balance-readers', action='store_true',
                        help='cap every reader to the size of the smallest '
                             'before splitting, so no single voice dominates '
                             '(see balance_by_reader)')
    parser.add_argument('--min-per-class', type=int, default=10,
                        help='drop classes with fewer than this many samples. '
                             'A stratified split needs at least 2 per class and '
                             'realistically ~10 to put any in train/val/test, so '
                             '--all-classes otherwise dies on the long tail: '
                             'BEN13R holds exactly 1 Yerach_Ben_Yomo and 1 '
                             'Qarney_Para, and sklearn refuses to stratify that. '
                             'Dropping them is honest -- a class with one example '
                             'cannot be learned or measured either way.')
    parser.add_argument('--exclude-align', nargs='+', default=None,
                        help="drop words whose meta 'align' field is in this list "
                             "before class selection/splitting -- e.g. "
                             "--exclude-align stored to drop the Whisper-boundary "
                             "fallback rows that verse-level CTC never replaced "
                             "(FINDINGS.md §8's pending boundary-quality ablation). "
                             "R1 has the highest 'stored' share of the three main "
                             "readers (36% vs R2's 23%/R8's 14%), so this is a "
                             "direct test of whether R1's wall is partly boundary "
                             "noise rather than purely voice/generalization.")
    parser.add_argument('--tag', default='',
                        help='suffix for the output dir. REQUIRED when the same '
                             'split is prepared from a different TAAMIFY_FEATURES '
                             'source: the scenario name is derived from '
                             'split/holdout/classes only, so without a tag a '
                             'stressed-window or speaker-normalised run silently '
                             'overwrites the word-window one of the same name, and '
                             'anything trained afterwards reads the wrong features '
                             'under the right-looking directory name.')
    args = parser.parse_args()

    if args.split == 'loro' and not args.holdout:
        parser.error("--split loro requires --holdout READER")

    print(f"{'='*60}\nLOADING READERS\n{'='*60}")
    X, taams, reader_ids, aligns, parashot = load_readers(args.readers)

    if args.exclude_align:
        keep = ~np.isin(aligns, args.exclude_align)
        print(f"\n--exclude-align {args.exclude_align}: dropping "
              f"{(~keep).sum()} of {len(keep)} words "
              f"({(~keep).sum() / len(keep) * 100:.1f}%)")
        X, taams, reader_ids = X[keep], taams[keep], reader_ids[keep]

    num_classes = None if args.all_classes else args.num_classes
    class_names = select_classes(taams, num_classes)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    if args.min_per_class > 1:
        from collections import Counter as _C
        counts = _C(taams)
        too_small = [c for c in class_names
                     if counts.get(c, 0) < args.min_per_class]
        if too_small:
            print(f"\nDropping {len(too_small)} class(es) below "
                  f"--min-per-class={args.min_per_class}: "
                  + ", ".join(f"{c}({counts.get(c,0)})" for c in too_small))
            class_names = [c for c in class_names if c not in too_small]
            label_to_idx = {name: i for i, name in enumerate(class_names)}

    X, taams, reader_ids, parashot, reasons = validate_and_filter(
        X, taams, reader_ids, class_names, parashot)
    y = np.array([label_to_idx[t] for t in taams], dtype=np.int64)

    if args.balance_readers:
        print("\nBalancing readers so no single voice dominates:")
        keep_before = len(X)
        X, y, reader_ids = balance_by_reader(X, y, reader_ids)
        if len(X) != keep_before:
            parashot = None   # indices no longer line up; parasha split is off

    print(f"\nNormalizing {len(X)} samples (per-sample zero-mean unit-var)")
    X = normalize_per_sample(X)
    X = X[:, None, :, :]  # (N, 1, 128, 87)

    if args.split == 'random':
        splits = make_random_splits(X, y, args.test_size, args.val_size)
        scenario = f"random_top{len(class_names)}"
    elif args.split == 'parasha':
        if parashot is None:
            raise SystemExit("--split parasha needs per-word parasha labels "
                             "(and is incompatible with --balance-readers)")
        splits = make_parasha_splits(X, y, parashot, args.test_parashot,
                                     args.val_size, draw_seed=args.parasha_seed)
        scenario = f"parasha{args.test_parashot}_top{len(class_names)}"
    else:
        splits = make_loro_splits(X, y, reader_ids, args.holdout, args.val_size,
                                  max_train=args.max_train)
        scenario = f"loro_{args.holdout}_top{len(class_names)}"
        if args.max_train:
            scenario += f"_cap{args.max_train}"

    # keeps feature-source variants in separate dirs -- see --tag's help
    if args.tag:
        scenario += f"_{args.tag}"

    X_train, y_train, X_val, y_val, X_test, y_test = splits
    print(f"\nSplit: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    output_dir = OUTPUT_ROOT / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'train.npz', X=X_train, y=y_train)
    np.savez_compressed(output_dir / 'val.npz', X=X_val, y=y_val)
    np.savez_compressed(output_dir / 'test.npz', X=X_test, y=y_test)

    meta = {
        'scenario': scenario,
        'split': args.split,
        'holdout': args.holdout,
        'readers': args.readers,
        'class_names': class_names,
        'label_to_idx': label_to_idx,
        'num_classes': len(class_names),
        'exclude_align': args.exclude_align,
        'input_shape': [1, N_MELS, MAX_FRAMES],
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'normalization': 'per_sample_zero_mean_unit_var',
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    generate_report(splits, class_names, reasons, output_dir, scenario)

    print(f"\n{'='*60}\nDONE -> {output_dir}\n{'='*60}")
    print(f"Next: python 10_train_pipeline.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
