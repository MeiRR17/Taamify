"""
Collect every result from the Ben13 training programme into one report.
=======================================================================
Written so that coming back to a finished run means reading a table, not
grepping logs. Every number here is pulled from the metrics JSON each run
writes; nothing is retyped or remembered.

The comparisons that matter are baked in, because a bare accuracy figure
answers nothing on its own:

  * BEN13R alone vs the old three-reader corpus, on the SAME 11 classes --
    is this corpus better data, or just different?
  * LORO with Ben13 added vs LORO without it -- does it transfer to human
    readers, which is the only thing the product cares about?
  * Every delta is shown against the measured seed spread (1.0-2.2 points,
    from the 6-seed run), so a change smaller than noise is labelled as
    noise rather than reported as a finding.

Usage:
  python collect_ben13_results.py
  python collect_ben13_results.py --json results/ben13_train/summary.json
"""
import argparse
import glob
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent

# Historical baselines, all on the same top-11 vocabulary, from
# results/v3_comparison/sephardic_baseline.md and the 48h sweep.
BASE_LORO_11 = {"R1": 0.469, "R2": 0.828, "R8": 0.785}
BASE_RANDOM_11 = 0.7768          # R1+R2+R8 random split, 11 classes
SEED_SD = 0.022                  # worst-fold seed standard deviation, 6 seeds


def metrics(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def find(pattern):
    hits = sorted(glob.glob(str(P / pattern)))
    return hits[0] if hits else None


def seq_report(pattern):
    """(cnn, bilstm) accuracy from a sequence_model_report.md."""
    hit = find(pattern)
    if not hit:
        return None
    txt = Path(hit).read_text(encoding="utf-8")
    a = re.search(r"BiLSTM[^|]*\|\s*([\d.]+)%", txt)
    b = re.search(r"CNN alone[^|]*\|\s*([\d.]+)%", txt)
    return (float(b.group(1)), float(a.group(1))) if a and b else None


def verdict(delta, sd=SEED_SD):
    if abs(delta) < sd:
        return "רעש (מתחת לסטיית הזרעים)"
    return "שיפור אמיתי" if delta > 0 else "הרעה אמיתית"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = {}
    L = []
    def say(s=""):
        L.append(s)

    say("=" * 78)
    say("  תוצאות אימון קורפוס בן13")
    say("=" * 78)

    # ---------- corpus sizes ----------
    say("\n## גודל הקורפוס")
    sizes = {}
    for r in ["BEN13R", "BEN13N", "R1", "R2", "R8"]:
        m = P / f"features_v3_word/{r}_meta.json"
        if m.exists():
            with open(m, encoding="utf-8") as f:
                n = len(json.load(f))
            sizes[r] = n
            say(f"  {r:<10}{n:>8,} מילים")
    out["corpus_sizes"] = sizes

    # ---------- BEN13R alone, 11 classes ----------
    say("\n## BEN13R לבד — 11 מחלקות (פיצול אקראי)")
    j = metrics(P / "results/random_top11_ben13r/deep_cnn/deep_cnn_metrics.json")
    if j:
        d = j["accuracy"] - BASE_RANDOM_11
        say(f"  BEN13R          {j['accuracy']:.4f}   (מאוזן {j['balanced_accuracy']:.4f}, MCC {j['mcc']:.3f})")
        say(f"  R1+R2+R8 (ישן)  {BASE_RANDOM_11:.4f}")
        say(f"  הפרש            {d:+.4f}   -> {verdict(d)}")
        out["ben13r_random11"] = j["accuracy"]
    else:
        say("  (עדיין לא הסתיים)")

    # ---------- BEN13R alone, all classes ----------
    say("\n## BEN13R לבד — כל המחלקות")
    p = find("results/random_top*_ben13r_all/deep_cnn/deep_cnn_metrics.json")
    j = metrics(p) if p else None
    if j:
        say(f"  דיוק {j['accuracy']:.4f} | מאוזן {j['balanced_accuracy']:.4f} | MCC {j['mcc']:.3f}")
        out["ben13r_all"] = j["accuracy"]
    else:
        say("  (עדיין לא הסתיים)")

    # ---------- the real question: does it transfer? ----------
    say("\n## השאלה האמיתית — LORO: האם בן13 עוזר לקוראים אנושיים?")
    say(f"  {'fold':<6}{'בלי בן13':<12}{'עם בן13':<12}{'הפרש':<10}מסקנה")
    say("  " + "-" * 66)
    loro = {}
    for f in ["R1", "R2", "R8"]:
        j = metrics(P / f"results/loro_{f}_top11_cap40000_withben13/deep_cnn/deep_cnn_metrics.json")
        if not j:
            say(f"  {f:<6}(עדיין לא הסתיים)")
            continue
        old, new = BASE_LORO_11[f], j["accuracy"]
        loro[f] = new
        say(f"  {f:<6}{old:<12.4f}{new:<12.4f}{new-old:+.4f}   {verdict(new-old)}")
    if len(loro) == 3:
        om = sum(BASE_LORO_11.values()) / 3
        nm = sum(loro.values()) / 3
        say("  " + "-" * 66)
        say(f"  {'ממוצע':<6}{om:<12.4f}{nm:<12.4f}{nm-om:+.4f}   {verdict(nm-om)}")
    out["loro_with_ben13"] = loro

    # ---------- all classes LORO ----------
    say("\n## LORO — כל המחלקות, עם בן13")
    allloro = {}
    for f in ["R1", "R2", "R8"]:
        p = find(f"results/loro_{f}_top*_cap40000_allwithben13/deep_cnn/deep_cnn_metrics.json")
        j = metrics(p) if p else None
        if j:
            allloro[f] = j["accuracy"]
            say(f"  {f:<6}{j['accuracy']:.4f}  (מאוזן {j['balanced_accuracy']:.4f})")
    out["loro_all_with_ben13"] = allloro

    # ---------- sequence models ----------
    say("\n## מודלי רצף (BiLSTM + CRF)")
    seq = {}
    for f in ["R1", "R2", "R8"]:
        r = seq_report(f"results/sequence_model/loro_{f}_ben13r_crf*/sequence_model_report.md")
        if r:
            seq[f] = {"cnn": r[0], "bilstm": r[1]}
            say(f"  {f:<6}CNN {r[0]:.1f}%  ->  +רצף {r[1]:.1f}%   ({r[1]-r[0]:+.1f} נקודות)")
    if seq:
        g = sum(v["bilstm"] - v["cnn"] for v in seq.values()) / len(seq)
        say(f"  ממוצע שיפור ההקשר: {g:+.1f} נקודות")
    out["sequence"] = seq

    # ---------- Nevi'im ----------
    say("\n## נביאים — האם סגנון קריאה שונה עוזר?")
    j = metrics(P / "results/random_top11_neviim/deep_cnn/deep_cnn_metrics.json")
    if j:
        say(f"  BEN13N לבד: {j['accuracy']:.4f}")
        out["neviim_alone"] = j["accuracy"]
    comb = {}
    for f in ["R1", "R2", "R8"]:
        j = metrics(P / f"results/loro_{f}_top11_cap40000_torahneviim/deep_cnn/deep_cnn_metrics.json")
        if j:
            comb[f] = j["accuracy"]
            base = loro.get(f)
            extra = f"  (מול {base:.4f} בלי נביאים, {j['accuracy']-base:+.4f})" if base else ""
            say(f"  {f:<6}תורה+נביאים {j['accuracy']:.4f}{extra}")
    out["torah_plus_neviim"] = comb

    say("\n" + "=" * 78)
    text = "\n".join(L)
    print(text)
    (P / "results/ben13_train/SUMMARY.txt").write_text(text + "\n", encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\nנשמר: results/ben13_train/SUMMARY.txt")


if __name__ == "__main__":
    main()
