"""
16_ben13_zeroshot_eval.py — does Stage A hold up on a voice it never trained on?
==================================================================================
The production Stage A verify-head (Milestone 5) was trained and measured on
R1/R2/R8 — the same three readers, just different verses/folds. Ben13's
segmented clips (Data_Prep/segment_ben13_clips.py, Milestone 1C) are a
genuinely independent zero-shot test: a fourth voice, studio-clean, that
never appears in training data at all.

Scope note: only 97 of the 114 segments have a true class inside the
production model's top-11 vocabulary (Tipecha, Munach, Mercha, Zaqef_Qatan,
Etnachta, Pashta, Siluk, Mahapakh, Tevir, Revia, Qadma) -- the other 17
(Darga, Zarqa, Segol, Geresh, Yerach_Ben_Yomo, Qarney_Para) have no slot in
the verify-head's one-hot space and are excluded, not silently dropped. Siluk
never appears (it's verse-final, doesn't occur mid-phrase in a combo clip).

Same pairing convention as 15_binary_verify_model.py's make_pairs(): every
segment gives one positive pair (its own true class) and one negative pair
(a random wrong class) -- so this measures the same thing Milestone 3 did,
just on audio the model has truly never seen.

Usage:
  python 16_ben13_zeroshot_eval.py
"""

import importlib.util
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_DIR = Path(__file__).parent
SEGMENTS = PROJECT_DIR / "data" / "ben13_segments" / "manifest.json"
OUT = PROJECT_DIR / "results" / "v3_comparison" / "ben13_zeroshot_eval.md"

CNN_CHECKPOINT = PROJECT_DIR / "results" / "random_top11" / "deep_cnn" / "deep_cnn_best.pth"
VERIFY_CHECKPOINT = PROJECT_DIR / "results" / "binary_verify" / "production_verify_head.pth"

SR = 22050
N_MELS = 128
FMAX = 8000
MAX_FRAMES = 87
PAD_DB = -80.0
SEED = 0


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def word_spectrogram(y):
    """Same spec as extract_features_v3.py / webapp/server.py's word_spectrogram,
    for a clip that's already isolated (no start/end trimming needed here)."""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])),
                        mode="constant", constant_values=PAD_DB)
    return mel_db[:, :MAX_FRAMES].astype(np.float32)


def main():
    train_mod = load_module("train_pipeline", "10_train_pipeline.py")
    bv_mod = load_module("binary_verify", "15_binary_verify_model.py")
    device = train_mod.pick_device()

    ckpt = torch.load(CNN_CHECKPOINT, map_location="cpu", weights_only=False)
    class_names = list(ckpt["class_names"])
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    cnn = train_mod.MODEL_REGISTRY[ckpt["architecture"]](num_classes=len(class_names))
    cnn.load_state_dict(ckpt["model_state_dict"])
    cnn.eval().to(device)

    v_ckpt = torch.load(VERIFY_CHECKPOINT, map_location="cpu", weights_only=False)
    assert list(v_ckpt["class_names"]) == class_names, \
        "verify-head class_names don't match the CNN's -- refusing to evaluate"
    head = bv_mod.VerifyHead(v_ckpt["emb_dim"], len(class_names), v_ckpt["hidden"])
    head.load_state_dict(v_ckpt["model_state_dict"])
    head.eval().to(device)

    with open(SEGMENTS, encoding="utf-8") as f:
        segments = json.load(f)
    usable = [e for e in segments if e["taam"] in label_to_idx]
    excluded = sorted({e["taam"] for e in segments if e["taam"] not in label_to_idx})
    print(f"{len(usable)}/{len(segments)} segments usable (excluded classes with no "
          f"slot in the top-11 verify-head: {', '.join(excluded)})")

    last_linear = [m for m in cnn.modules() if isinstance(m, nn.Linear)][-1]
    captured = []
    handle = last_linear.register_forward_hook(
        lambda _m, inp, _out: captured.append(inp[0].detach().cpu()))

    emb_list, y_list, low_conf_list = [], [], []
    with torch.no_grad():
        for e in usable:
            y_audio, _ = librosa.load(str(PROJECT_DIR / e["local_path"]), sr=SR, mono=True)
            spec = word_spectrogram(y_audio)
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            x = torch.from_numpy(spec)[None, None].to(device)
            cnn(x)
            emb_list.append(captured[-1][0].numpy())
            y_list.append(label_to_idx[e["taam"]])
            low_conf_list.append(e["low_confidence"])
    handle.remove()

    emb = np.stack(emb_list)
    y = np.array(y_list)
    low_conf = np.array(low_conf_list)

    Xp, Cp, Mp = bv_mod.make_pairs(emb, y, len(class_names), SEED)

    # make_pairs() draws `wrong` then permutes, both from one continuing rng
    # stream -- replicate that exact sequence (not a fresh rng for the permute
    # step) so this permutation lines up with Xp/Cp/Mp's order exactly.
    n = len(y)
    rng = np.random.default_rng(SEED)
    wrong = rng.integers(0, len(class_names) - 1, size=n)
    wrong = wrong + (wrong >= y)
    perm = rng.permutation(2 * n)
    low_conf_pairs = np.concatenate([low_conf, low_conf])[perm]

    # self-check: this reconstructed perm must reproduce make_pairs()'s own
    # Cp/Mp exactly, or the low-confidence mask below is silently misaligned
    cand_check = np.concatenate([y, wrong])[perm].astype(np.int64)
    match_check = np.concatenate([np.ones(n), np.zeros(n)])[perm].astype(np.float32)
    assert np.array_equal(cand_check, Cp) and np.array_equal(match_check, Mp), \
        "permutation reconstruction does not match make_pairs() -- low_conf_pairs " \
        "would be misaligned, refusing to report a result"

    with torch.no_grad():
        xb = torch.from_numpy(Xp).float().to(device)
        cb = torch.zeros(len(Cp), len(class_names), device=device)
        cb[torch.arange(len(Cp)), torch.from_numpy(Cp)] = 1.0
        probs = torch.sigmoid(head(xb, cb)).cpu().numpy()
    preds = (probs >= 0.5).astype(np.float32)

    def report(mask, label):
        m, p, pr = Mp[mask], preds[mask], probs[mask]
        acc = accuracy_score(m, p)
        prec = precision_score(m, p, zero_division=0)
        rec = recall_score(m, p, zero_division=0)
        f1 = f1_score(m, p, zero_division=0)
        pos_conf = pr[m == 1]
        neg_conf = pr[m == 0]
        return {
            "label": label, "n_pairs": int(mask.sum()),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "pos_conf_mean": float(pos_conf.mean()) if len(pos_conf) else float("nan"),
            "pos_conf_median": float(np.median(pos_conf)) if len(pos_conf) else float("nan"),
            "neg_conf_mean": float(neg_conf.mean()) if len(neg_conf) else float("nan"),
            "neg_conf_median": float(np.median(neg_conf)) if len(neg_conf) else float("nan"),
        }

    all_mask = np.ones(len(Mp), dtype=bool)
    hc_mask = ~low_conf_pairs
    results = [report(all_mask, "All 97 segments (194 pairs)"),
               report(hc_mask, f"High-confidence only ({int(hc_mask.sum()//2)} segments, "
                                f"{int(hc_mask.sum())} pairs)")]

    print()
    for r in results:
        print(f"{r['label']}: acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
              f"rec={r['recall']:.3f} f1={r['f1']:.3f} | "
              f"match conf mean/median={r['pos_conf_mean']:.3f}/{r['pos_conf_median']:.3f} "
              f"| no-match conf mean/median={r['neg_conf_mean']:.3f}/{r['neg_conf_median']:.3f}")

    lines = [
        "# Ben13 zero-shot evaluation — Stage A on a voice never in training",
        "",
        "The production Stage A verify-head (Milestone 5) trained and measured on "
        "R1/R2/R8. Ben13's segmented clips (`Data_Prep/segment_ben13_clips.py`, "
        "Milestone 1C) are a genuinely independent test: a fourth voice, studio-clean, "
        "that appears nowhere in training.",
        "",
        f"**Scope**: {len(usable)}/{len(segments)} segments have a true class inside "
        f"the production model's top-11 vocabulary; the other {len(segments) - len(usable)} "
        f"({', '.join(excluded)}) have no slot in the verify-head's one-hot space and are "
        "excluded, not silently dropped. Siluk never appears (verse-final, doesn't occur "
        "mid-phrase in a combo clip). Same pairing convention as Milestone 3: every "
        "segment gives one positive pair (true class) and one random-wrong-class negative.",
        "",
        "| Set | Pairs | Accuracy | Precision | Recall | F1 | Match conf. (mean/median) | No-match conf. (mean/median) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['label']} | {r['n_pairs']} | {r['accuracy']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['pos_conf_mean']:.3f} / {r['pos_conf_median']:.3f} | "
                     f"{r['neg_conf_mean']:.3f} / {r['neg_conf_median']:.3f} |")
    lines += [
        "",
        f"For reference, Milestone 3's R1/R2/R8 LORO mean was **88.4%** (range 77.8-94.4%).",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
