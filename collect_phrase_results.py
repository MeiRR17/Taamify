"""
Collect the phrase-realignment experiment into one report.
============================================================
Two questions, both answered directly against a metrics.json rather than
retyped:

  1. Per reader alone, random split: does re-aligning in phrase-sized chunks
     (Ben13's structural advantage) improve accuracy on the SAME reader's
     OWN voice? Isolates alignment quality from cross-reader generalisation.

  2. LORO, holding out each reader's ORIGINAL alignment (so the test set is
     identical to every other number in this project) while training on the
     OTHER two readers' phrase-realigned versions: does the R1 wall move?

Usage:
  python collect_phrase_results.py
"""
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
BASE_LORO_11 = {"R1": 0.469, "R2": 0.828, "R8": 0.785}
SEED_SD = 0.022


def m(path):
    try:
        with open(P / path) as f:
            return json.load(f)
    except Exception:
        return None


def verdict(d, sd=SEED_SD):
    if abs(d) < sd:
        return "רעש"
    return "שיפור אמיתי" if d > 0 else "הרעה אמיתית"


def main():
    L = []
    def say(s=""): L.append(s)

    say("=" * 70)
    say("  תוצאות: יישור מחדש ברמת קטע (Ben13-style) עבור R1/R2/R8")
    say("=" * 70)

    say("\n## לכל קורא לבד — מקורי מול קטע-מוחדש (פיצול אקראי, 11 טעמים)")
    say(f"  {'קורא':<6}{'מקורי':<12}{'קטע-מוחדש':<14}{'הפרש':<10}מסקנה")
    say("  " + "-" * 62)
    for r in ["R1", "R2", "R8"]:
        a = m(f"results/random_top11_orig_{r}/deep_cnn/deep_cnn_metrics.json")
        b = m(f"results/random_top11_phrase_{r}/deep_cnn/deep_cnn_metrics.json")
        if not (a and b):
            say(f"  {r:<6}(עדיין לא הסתיים)")
            continue
        d = b["accuracy"] - a["accuracy"]
        say(f"  {r:<6}{a['accuracy']:<12.4f}{b['accuracy']:<14.4f}{d:+.4f}   {verdict(d)}")

    say("\n## השאלה האמיתית — האם הקיר של R1 זז?")
    say("  (בדיקה בכל fold: מתאמן על שני הקוראים האחרים, קטע-מוחדשים;")
    say("   נבדק על הקורא השלישי, ההתראה המקורית שלו — אותו מבחן כמו בכל מספר אחר בפרויקט)")
    say(f"  {'fold':<6}{'ישן (LORO רגיל)':<18}{'קטע-מוחדש':<14}{'הפרש':<10}מסקנה")
    say("  " + "-" * 66)
    vals = []
    for f in ["R1", "R2", "R8"]:
        j = m(f"results/loro_{f}_top11_cap40000_phraseloro/deep_cnn/deep_cnn_metrics.json")
        if not j:
            say(f"  {f:<6}(עדיין לא הסתיים)")
            continue
        old = BASE_LORO_11[f]
        new = j["accuracy"]
        d = new - old
        vals.append((old, new))
        say(f"  {f:<6}{old:<18.4f}{new:<14.4f}{d:+.4f}   {verdict(d)}")
    if len(vals) == 3:
        om = statistics.mean(v[0] for v in vals)
        nm = statistics.mean(v[1] for v in vals)
        say("  " + "-" * 66)
        say(f"  {'ממוצע':<6}{om:<18.4f}{nm:<14.4f}{nm-om:+.4f}   {verdict(nm-om)}")

    say("\n" + "=" * 70)
    text = "\n".join(L)
    print(text)
    (P / "results/phrase_realign/SUMMARY.txt").write_text(text + "\n", encoding="utf-8")
    print("\nנשמר: results/phrase_realign/SUMMARY.txt")


if __name__ == "__main__":
    main()
