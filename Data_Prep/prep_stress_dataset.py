"""
Station 4, measured cheaply: is stress position audible at all?
================================================================
The roadmap puts syllable-level alignment (station 3) before stress detection
(station 4), on the assumption that judging WHICH syllable is stressed needs
syllable boundaries. That is probably true for the final product -- but it is
the expensive way to find out whether the signal exists.

The cheap way first: the whole-word spectrograms are ALREADY extracted
(features_v3_word/BEN13R_X.npy), and the stress label is ALREADY derivable
from the ta'am's position (Data_Prep/stressed_syllable.py, 100% coverage).
So the same array can simply be relabelled: instead of "which ta'am", ask
"milra or mil'el". No new extraction, no new alignment, no new recordings --
one training run answers whether stress is acoustically recoverable.

If balanced accuracy lands well above the majority baseline, station 3 is
worth building. If it doesn't, station 3 would have been weeks spent on a
signal that isn't there.

MEASURED BASELINE THAT MATTERS
------------------------------
81.3% of words are milra (final stress). A model that always answers "milra"
therefore scores 81.3% and has learned nothing. Plain accuracy is useless
here; **balanced accuracy** (the average of per-class recall) is the honest
metric, and a trivial model scores 50% on it.

Monosyllables are excluded -- they carry no stress question at all.

Usage:
  python Data_Prep/prep_stress_dataset.py --split parasha
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stressed_syllable import stressed_syllable   # noqa: E402

FEAT = PROJECT_DIR / "features_v3_word"
OUT_ROOT = PROJECT_DIR / "prepared_data"
CLASSES = ["milra", "milel"]          # final stress, penultimate stress

BGD = set("בגדכפת")
DAGESH = "\u05bc"


def hebrew_letters(w):
    return [c for c in w if "\u05d0" <= c <= "\u05ea"]


def label_for(task, w):
    """Return (label, class_names) or None when the word cannot carry the
    distinction at all. Words are restricted to those containing EXACTLY ONE
    instance of the relevant letter, so a word-level label is unambiguous
    about which sound the model has to find."""
    if task == "stress":
        return None            # handled separately, needs syllable analysis
    L = hebrew_letters(w)
    if task == "dagesh":
        b = [c for c in L if c in BGD]
        if len(b) != 1:
            return None
        i = w.index(b[0])
        return 1 if DAGESH in w[i:i + 3] else 0
    if task == "guttural":
        a = [c for c in L if c in "\u05e2\u05d0"]
        if len(a) != 1:
            return None
        return 1 if a[0] == "\u05e2" else 0
    return None


CLASS_NAMES = {
    "stress": ["milra", "milel"],
    "dagesh": ["no_dagesh", "dagesh"],
    "guttural": ["alef", "ayin"],
}

N_MELS, MAX_FRAMES, PAD_DB = 128, 87, -80.0


def validate(spec):
    if spec.shape != (N_MELS, MAX_FRAMES):
        return False
    if np.isnan(spec).any() or np.isinf(spec).any():
        return False
    pad = np.all(spec <= PAD_DB + 0.1, axis=0) | np.all(spec == 0, axis=0)
    return pad.mean() <= 0.85 and spec.std() >= 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="BEN13R")
    ap.add_argument("--task", choices=["stress", "dagesh", "guttural"],
                    default="stress",
                    help="stress: milra vs milel (station 4). "
                         "dagesh: does the word's single begadkefat letter carry "
                         "a dagesh kal -- i.e. can the model hear b vs v, k vs kh, "
                         "p vs f (station 5). "
                         "guttural: ayin vs alef -- the distinction that marks "
                         "the Sephardic-Jerusalem pronunciation and that most "
                         "Israeli speakers collapse (station 5)")
    ap.add_argument("--split", choices=["random", "parasha"], default="parasha")
    ap.add_argument("--test-parashot", type=int, default=8)
    ap.add_argument("--tag", default="stress")
    args = ap.parse_args()

    X = np.load(FEAT / f"{args.reader}_X.npy").astype(np.float32)
    meta = json.loads((FEAT / f"{args.reader}_meta.json").read_text(encoding="utf-8"))
    assert len(X) == len(meta)
    print(f"{args.reader}: {len(X):,} words")

    keep, ys, par, reasons = [], [], [], Counter()
    for i, m in enumerate(meta):
        w = m.get("word_raw")
        if not w:
            reasons["no_pointed_text"] += 1
            continue
        if args.task == "stress":
            try:
                s = stressed_syllable(w)
            except Exception:
                s = None
            if not s or not s.get("ok"):
                reasons["stress_undetermined"] += 1
                continue
            if s["n_syllables"] < 2:
                reasons["monosyllable"] += 1
                continue
            fe = s["from_end"]
            if fe not in (1, 2):
                reasons["stress_further_back"] += 1
                continue
            lab = 0 if fe == 1 else 1
        else:
            lab = label_for(args.task, w)
            if lab is None:
                reasons["letter_not_unique"] += 1
                continue
        if not validate(X[i]):
            reasons["bad_spectrogram"] += 1
            continue
        keep.append(i)
        ys.append(lab)
        par.append(str(m.get("parasha", "?")))

    keep = np.array(keep)
    y = np.array(ys, dtype=np.int64)
    par = np.array(par)
    print(f"usable: {len(keep):,}")
    for r, n in reasons.most_common():
        print(f"  dropped {r}: {n:,}")
    c = Counter(y.tolist())
    maj = max(c.values()) / len(y) * 100
    print(f"\nlabels: milra={c[0]:,}  milel={c[1]:,}")
    print(f"majority-class baseline: {maj:.1f}% accuracy, 50.0% balanced")

    Xk = X[keep]
    mean = Xk.mean(axis=(1, 2), keepdims=True)
    std = Xk.std(axis=(1, 2), keepdims=True); std[std == 0] = 1
    Xk = ((Xk - mean) / std)[:, None, :, :]

    if args.split == "parasha":
        rng = np.random.default_rng(42)
        uniq = np.unique(par)
        test_p = set(rng.choice(uniq, size=args.test_parashot, replace=False).tolist())
        tm = np.isin(par, list(test_p))
        Xte, yte = Xk[tm], y[tm]
        Xr, yr = Xk[~tm], y[~tm]
        Xtr, Xva, ytr, yva = train_test_split(Xr, yr, test_size=.15,
                                              random_state=42, stratify=yr)
        scen = f"{args.task}{args.test_parashot}_{args.tag}"
        print(f"\nheld out {args.test_parashot} whole parashot: {sorted(test_p)}")
    else:
        Xtv, Xte, ytv, yte = train_test_split(Xk, y, test_size=.15,
                                              random_state=42, stratify=y)
        Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=.176,
                                              random_state=42, stratify=ytv)
        scen = f"{args.task}random_{args.tag}"

    out = OUT_ROOT / scen
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train.npz", X=Xtr, y=ytr)
    np.savez_compressed(out / "val.npz", X=Xva, y=yva)
    np.savez_compressed(out / "test.npz", X=Xte, y=yte)
    (out / "metadata.json").write_text(json.dumps({
        "scenario": scen, "class_names": CLASS_NAMES[args.task], "num_classes": 2,
        "input_shape": [1, N_MELS, MAX_FRAMES],
        "train_samples": len(Xtr), "val_samples": len(Xva), "test_samples": len(Xte),
        "majority_baseline_accuracy": maj,
        "normalization": "per_sample_zero_mean_unit_var",
        "note": "labels are stress position derived from the ta'am's own "
                "placement; balanced accuracy is the only honest metric here",
    }, indent=2), encoding="utf-8")
    print(f"train {len(Xtr):,}  val {len(Xva):,}  test {len(Xte):,}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
