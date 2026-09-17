"""
The product-level noise question, asked through the product's own path.
======================================================================
`eval_noise_robustness.py` raises a noise floor on the mel-spectrogram. That
is the same operation training's --noise performs, so it is the right test
for "does augmentation help the classifier" -- but it cannot see the failure
that actually breaks a practice session in a real room, because it starts
*after* the audio has already been turned into a clean word window.

In a real room the damage happens earlier:

  1. the CTC aligner has to find word boundaries in noisy audio, and when it
     fails there are no windows to classify at all
  2. the windows it does find are shifted, so the crop is wrong
  3. only then does the classifier see a degraded spectrogram

So this adds noise to the WAVEFORM, at honest SNRs, and pushes it through
/api/check_phrase -- the same endpoint the app calls. What comes back is the
number that matters: in a room at this SNR, how often does the app tell a
learner who read CORRECTLY that they were wrong?

Noise is pink (1/f), not white: room tone, traffic and fans are all heavily
low-frequency, and white noise would flatter the system by leaving the low
mel bins -- where most cantillation energy sits -- untouched.

Usage:
  python Data_Prep/eval_room_noise.py --phrases 40
"""
import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
READING = PROJECT_DIR / "data" / "ben13_reading"
OUT = PROJECT_DIR / "results" / "noise_robustness"
API = "http://127.0.0.1:8000"
SNRS = [None, 20, 15, 10, 5]     # None = clean original


def pink_noise(n, rng):
    """1/f noise via spectral shaping -- the shape of room tone, traffic and
    fans. White noise would leave the low mel bins alone, which is where most
    of the cantillation energy is, and so would understate the damage."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / 16000)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    spec *= shape
    out = np.fft.irfft(spec, n=n)
    return out / (np.abs(out).max() + 1e-9)


def mix_at_snr(y, snr_db, rng):
    if snr_db is None:
        return y
    noise = pink_noise(len(y), rng)
    sig_p = float(np.mean(y ** 2)) + 1e-12
    noise_p = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(sig_p / (noise_p * (10 ** (snr_db / 10.0))))
    mixed = y + noise * scale
    peak = np.abs(mixed).max()
    return mixed / peak * 0.95 if peak > 1.0 else mixed


def pick_phrases(limit):
    """Phrases from Ben13's own reading corpus, with the ta'amim the text
    requires. Ben13 cuts at disjunctive accents, so one audio block IS one
    practice phrase -- no re-segmentation, no alignment guesswork."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tg", PROJECT_DIR / "Data_Prep" / "taam_grammar.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)

    rows = json.loads((READING / "maintext.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    out = []
    for r in rows:
        if str(r.get("block")) in ("-1", "0") or not r.get("url"):
            continue
        path = READING / "yerushalmi" / r["url"].replace("01-BERESHIT", "01-bereshit")
        if not path.exists():
            cands = list((READING / "yerushalmi").glob(f"*/{r['url'].split('/', 1)[1]}")) \
                if "/" in r["url"] else []
            if not cands:
                continue
            path = cands[0]
        toks = tg.verse_tokens(r["text"])
        toks = [(w, t) for w, t in toks if t]
        if len(toks) < 2:
            continue
        out.append({"path": str(path),
                    "words": [w for w, _ in toks],
                    "taamim": [t for _, t in toks]})
        if len(out) >= limit:
            break
    return out


def check(path, words, taamim):
    cmd = ["curl", "-s", "-X", "POST", f"{API}/api/check_phrase",
           "-F", f"audio=@{path}",
           "-F", f"words={json.dumps(words, ensure_ascii=False)}",
           "-F", f"taamim={json.dumps(taamim)}"]
    try:
        return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except (ValueError, subprocess.SubprocessError):
        return {}


def main():
    import soundfile as sf
    import librosa

    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", type=int, default=40)
    args = ap.parse_args()

    phrases = pick_phrases(args.phrases)
    print(f"{len(phrases)} phrases from Ben13's own reading corpus\n")
    rng = np.random.default_rng(7)

    L = []
    def say(s=""):
        L.append(s); print(s, flush=True)

    say("=" * 72)
    say("  Room noise, measured through /api/check_phrase (the app's own path)")
    say("=" * 72)
    say("\n  Every reading below is CORRECT -- it is Ben13 reading his own text.")
    say("  So every 'wrong' verdict is the expensive error: the app")
    say("  contradicting a learner who read it right.\n")
    say(f"  {'SNR':>8}{'phrases aligned':>18}{'words judged':>15}"
        f"{'CONTRADICTED':>15}")
    say("  " + "-" * 56)

    results = {}
    for snr in SNRS:
        aligned = total = judged = contradicted = 0
        for ph in phrases:
            total += 1
            y, sr = librosa.load(ph["path"], sr=16000, mono=True)
            if len(y) < 1600:
                continue
            noisy = mix_at_snr(y, snr, rng)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, noisy, 16000)
                d = check(f.name, ph["words"], ph["taamim"])
            Path(f.name).unlink(missing_ok=True)
            if not d.get("aligned"):
                continue
            aligned += 1
            for w in d.get("words", []):
                if w["status"] in ("ok", "wrong"):
                    judged += 1
                    contradicted += 1 if w["status"] == "wrong" else 0
        rate = contradicted / judged if judged else float("nan")
        label = "clean" if snr is None else f"{snr} dB"
        say(f"  {label:>8}{f'{aligned}/{total}':>18}{judged:>15}{rate*100:>14.1f}%")
        results[label] = {"aligned": aligned, "of": total, "judged": judged,
                          "contradicted": contradicted, "rate": rate}

    say("\n  A quiet room is about 35-45 dB SNR for a close mic; 15-20 dB is a")
    say("  fan and a street outside; 5-10 dB is a sibling talking nearby.")
    say("\n" + "=" * 72)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "room_noise_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "room_noise.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT}/room_noise.json")


if __name__ == "__main__":
    main()
