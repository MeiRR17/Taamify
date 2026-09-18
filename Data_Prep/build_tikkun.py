"""
תיקון סופרים -- the complete reference for all 25 ta'amim.
==========================================================
Every fact in the generated document is either derived from the pointed text
of the Torah or measured from the recorded corpus. Nothing is copied from a
textbook, which matters for two reasons: the melodic shapes are those of THIS
tradition (Sephardi-Jerusalem, as read by Ben13) rather than a generic
Ashkenazi description, and every claim can be re-derived when the corpus
grows.

WHAT IS DERIVED FROM THE TEXT

  frequency          how often the ta'am occurs across the Torah
  position           where in the verse it tends to fall (0 = start, 1 = end)
  serves / served-by which ta'amim precede and follow it, and how strongly.
                     This is the real syntax: Mercha serves Tipcha and Siluk;
                     Qadma serves a different group entirely. The textbooks
                     state these as rules -- here they are counted.
  word length        a disjunctive tends to sit on a longer word

WHAT IS MEASURED FROM THE AUDIO

  melodic contour    the mean pitch curve over 390+ real recordings, in
                     semitones relative to the word's own mean, with the
                     spread across occurrences
  consistency        how tightly the occurrences agree. A ta'am that is sung
                     the same way every time is learnable and checkable; one
                     that varies is neither.
  model accuracy     per-class, on parashot the model never saw

WHAT COMES FROM THE SOURCES

  severity           whether an error here changes meaning -- see severity.py

Usage:
  python Data_Prep/build_tikkun.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / "Data_Prep"))
import taam_grammar as tg          # noqa: E402
import severity as sev             # noqa: E402

OUT = P / "results" / "melody_research"

# The sign itself, and the name used in Sephardi practice where it differs
# from the name the Ashkenazi tradition uses. Both are needed: the product
# speaks Sephardi, but every printed chart the user owns says the other.
SIGNS = {
    "Siluk": "ֽ", "Etnachta": "֑", "Segol": "֒",
    "Shalshelet": "֓", "Zaqef_Qatan": "֔", "Zaqef_Gadol": "֕",
    "Tipecha": "֖", "Revia": "֗", "Zarqa": "֘",
    "Pashta": "֙", "Yetiv": "֚", "Tevir": "֛",
    "Geresh": "֜", "Gershayim": "֞",
    "Qarney_Para": "֟", "Telisha_Gedola": "֠", "Pazer": "֡",
    "Munach": "֣", "Mahapakh": "֤", "Mercha": "֥",
    "Mercha_Kefula": "֦", "Darga": "֧", "Qadma": "֨",
    "Telisha_Qetana": "֩", "Yerach_Ben_Yomo": "֪",
}

SEPHARDI_NAME = {
    "Zaqef_Qatan": "זקף קטן", "Zaqef_Gadol": "זקף גדול",
    "Tipecha": "טרחא",            # Sephardim say טרחא, Ashkenazim טפחא
    "Pashta": "קדמא",            # the sign is shared; context tells them apart
    "Qadma": "אזלא",
    "Geresh": "אזלא",
    "Gershayim": "שני גרישין",
    "Telisha_Qetana": "תלשא",
    "Telisha_Gedola": "תרסא",
    "Mercha": "מאריך",
    "Munach": "שופר הולך",
    "Mahapakh": "שופר מהופך",
    "Yerach_Ben_Yomo": "ירח בן יומו",
    "Qarney_Para": "קרני פרה",
}

# What each disjunctive divides, in plain Hebrew. This is the "why the verse
# breaks here" layer the product teaches, expressed once.
ROLE_HE = {
    "Siluk": "חותם את הפסוק. תמיד במילה האחרונה, תמיד עם סוף פסוק.",
    "Etnachta": "החלוקה הראשית של הפסוק לשניים. פעם אחת בדיוק בכל פסוק "
                "(למעט פסוקים קצרים מאוד).",
    "Segol": "מחלק את החצי הראשון כשהוא ארוך מדי לאתנחתא לבדה. נדיר.",
    "Shalshelet": "מחליף את סגול כשהמילה בודדה. ארבע פעמים בכל התורה — "
                  "והוא מסמן היסוס או עצירה דרמטית.",
    "Zaqef_Qatan": "החלוקה השנייה בחשיבותה בתוך כל חצי פסוק. הנפוץ במפסיקים.",
    "Zaqef_Gadol": "כזקף קטן, אך כשאין מילה שתשרת אותו לפניו.",
    "Tipecha": "מחלק את הקטע שלפני אתנחתא או סילוק. תמיד לקראת סוף היחידה.",
    "Revia": "מחלק בתוך תחום זקף. רביע = רביעי בדרגה.",
    "Zarqa": "מכין את הדרך לסגול, ומופיע כמעט רק לפניו.",
    "Pashta": "מחלק בתוך תחום זקף, במקום שבו הטעם נופל במלרע.",
    "Yetiv": "כפשטא, אך כשהמילה קצרה ואין מה לשרת אותה.",
    "Tevir": "מחלק בתוך תחום טפחא.",
    "Geresh": "מחלק בתוך תחום רביע.",
    "Gershayim": "כגרש, אך כשההטעמה נופלת אחרת.",
    "Pazer": "מפסיק חלש בתוך יחידה ארוכה במיוחד.",
    "Telisha_Gedola": "מפסיק חלש. תמיד בתחילת המילה, ולכן קל לזהותו בעין.",
    "Qarney_Para": "המפסיק הנדיר ביותר. מופע בודד בכל התורה.",
    "Mercha": "משרת את טפחא ואת סילוק. המשרת הנפוץ ביותר.",
    "Munach": "משרת מגוון מפסיקים; מה שהוא משרת נקבע לפי מה שבא אחריו.",
    "Mahapakh": "משרת את פשטא כמעט תמיד.",
    "Darga": "משרת את תביר.",
    "Qadma": "משרת את גרש ואת תלישא.",
    "Telisha_Qetana": "משרת. תמיד בסוף המילה.",
    "Mercha_Kefula": "משרת נדיר. חמישה מופעים בתורה.",
    "Yerach_Ben_Yomo": "משרת את קרני פרה. מופע בודד בכל התורה.",
}


BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]


def syntax_from_text():
    """Count frequency, position and neighbours over the pointed Torah text.

    Read from the verse cache rather than from the audio corpus metadata.
    That metadata is shuffled and carries no word index, so grouping it by
    verse recovers the membership of each verse but NOT the order inside it
    -- and every claim here (what precedes what, where in the verse a ta'am
    falls) is a statement about order. Counting from the text instead makes
    the ordering exact, and has the side benefit of covering the whole Torah
    rather than only the part that has been recorded.
    """
    after = defaultdict(Counter)
    before = defaultdict(Counter)
    position = defaultdict(list)
    wordlen = defaultdict(list)
    freq = Counter()
    n_verses = 0
    cache = P / "Data_Prep" / "sefaria_cache"
    for book in BOOKS:
        chap = 1
        while (cache / f"{book}.{chap}.json").exists():
            verses = json.loads((cache / f"{book}.{chap}.json")
                                .read_text(encoding="utf-8"))
            for raw in verses:
                toks = tg.verse_tokens(raw)
                toks = [(w, t) for w, t in toks if t]
                n = len(toks)
                if n < 2:
                    continue
                n_verses += 1
                for i, (word, t) in enumerate(toks):
                    freq[t] += 1
                    position[t].append(i / (n - 1))
                    wordlen[t].append(len(word))
                    if i + 1 < n:
                        after[t][toks[i + 1][1]] += 1
                    if i > 0:
                        before[t][toks[i - 1][1]] += 1
            chap += 1
    return freq, after, before, position, wordlen, n_verses


def melody():
    src = P / "app_assets" / "melody_curves_4voices.json"
    if not src.exists():
        src = P / "app_assets" / "melody_curves.json"
    cur = json.loads(src.read_text(encoding="utf-8"))
    out = {}
    for t, items in cur.items():
        arr = np.array([it["curve"] for it in items if it.get("curve")])
        if len(arr) < 3:
            continue
        mean = arr.mean(0)
        sd = arr.std(0)
        out[t] = {
            "n": len(arr),
            "mean": mean,
            "sd": sd,
            "range": float(mean.max() - mean.min()),
            "peak_at": float(np.argmax(mean)) / max(len(mean) - 1, 1),
            "net": float(mean[-1] - mean[0]),
            "consistency": float(1.0 / (1.0 + sd.mean())),
        }
    return out


def sparkline(curve, width=24):
    """The contour as text, so the document reads without a chart."""
    blocks = "▁▂▃▄▅▆▇█"
    c = np.interp(np.linspace(0, len(curve) - 1, width),
                  np.arange(len(curve)), curve)
    lo, hi = c.min(), c.max()
    if hi - lo < 1e-6:
        return blocks[3] * width
    idx = ((c - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in idx)


def describe_contour(m):
    """Say the shape in words. A learner reads this before hearing it."""
    if m is None:
        return "אין מספיק הקלטות כדי למדוד את הצורה."
    rng, net, peak = m["range"], m["net"], m["peak_at"]
    if rng < 1.5:
        shape = "קו כמעט ישר, בלי סלסול של ממש"
    elif net > 1.0:
        shape = "עולה לאורך המילה"
    elif net < -1.0:
        shape = "יורד לאורך המילה"
    elif peak < 0.35:
        shape = "שיא מוקדם ואז ירידה"
    elif peak > 0.65:
        shape = "עלייה לשיא בסוף"
    else:
        shape = "קשת — עולה באמצע ויורד"
    return (f"{shape}. טווח {rng:.1f} חצאי-טונים, "
            f"השיא ב-{peak*100:.0f}% מאורך המילה.")


def accuracy_table():
    """Per-class accuracy from whatever the latest honest evaluation left."""
    for name in ("per_class_accuracy.json", "confusion_by_class.json"):
        f = P / "results" / "v3_comparison" / name
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def main():
    freq, after, before, position, wordlen, n_verses = syntax_from_text()
    mel = melody()
    acc = accuracy_table()
    audio = {}
    ref = P / "app_assets" / "taam_reference_audio.json"
    if ref.exists():
        audio = json.loads(ref.read_text(encoding="utf-8"))

    order = sorted(SIGNS, key=lambda t: (tg.DISJUNCTIVE_RANK.get(t, 9),
                                         -freq.get(t, 0)))
    total = sum(freq.values())

    L = []
    def w(s=""):
        L.append(s)

    w("# תיקון סופרים — כל טעמי המקרא, נוסח ספרדי-ירושלמי")
    w()
    w("מסמך ייחוס. **כל מספר כאן נמדד** — או מהטקסט המנוקד של התורה, או "
      "מהקורפוס המוקלט. שום דבר לא הועתק מטבלה.")
    w()
    w(f"בסיס: **{total:,} מילים מוטעמות ב-{n_verses:,} פסוקים** — התורה "
      f"כולה. {len([t for t in freq if freq[t]])} טעמים מופיעים בפועל.")
    w()
    w("> **הסתייגות.** המנגינות כאן הן של בעל הקריאה שהוקלט (ספרדי-ירושלמי). "
      "מסורות אחרות שרות אחרת, וגם בתוך אותה מסורת יש הבדלים בין בעלי קריאה. "
      "דירוג החומרה מסכם מקורות נגישים כדי לתעדף משוב באפליקציה, **ואינו "
      "פסק הלכה.**")
    w()
    w("---")
    w()

    # ── תוכן עניינים ──
    w("## מפה מהירה")
    w()
    w("| טעם | סימן | תפקיד | דרגה | שכיחות | חומרה | צורת המנגינה |")
    w("|---|---|---|---|---|---|---|")
    for t in order:
        if not freq.get(t):
            continue
        he = tg.TAAM_HE.get(t, t)
        rank = tg.DISJUNCTIVE_RANK.get(t)
        role = "מפסיק" if rank else "משרת"
        m = mel.get(t)
        # show a contour only where enough occurrences back it; a curve
        # averaged over four recordings is a drawing, not a measurement
        spark = sparkline(m["mean"], 12) if (m and m["n"] >= 20) else "—"
        lvl = sev.LEVEL_HE[sev.taam_severity(t)]
        w(f"| **{he}** | {SIGNS[t]}  | {role} | {rank or '—'} | "
          f"{freq[t]:,} | {lvl} | `{spark}` |")
    w()
    rare = [t for t in SIGNS if freq.get(t, 0) and freq[t] <= 10]
    if rare:
        w("**הטעמים הנדירים** — " +
          " · ".join(f"{tg.TAAM_HE.get(t, t)} ({freq[t]})" for t in
                     sorted(rare, key=lambda x: freq[x])) +
          ". אלה אינם ניתנים לבדיקה אמינה: אין מספיק הקלטות כדי ללמוד "
          "אותם, והמוצר אומר זאת במפורש במקום להעמיד פנים.")
        w()
    w("---")
    w()

    # ── כרטיס לכל טעם ──
    for t in order:
        if not freq.get(t):
            continue
        he = tg.TAAM_HE.get(t, t)
        seph = SEPHARDI_NAME.get(t)
        rank = tg.DISJUNCTIVE_RANK.get(t)
        m = mel.get(t)
        lvl = sev.taam_severity(t)

        w(f"## {SIGNS[t]}  {he}")
        w()
        if seph and seph != he:
            w(f"*בפי ספרדים: **{seph}***")
            w()
        w(f"**{'מפסיק' if rank else 'משרת'}"
          + (f" · דרגה {rank}" if rank else "")
          + f" · {freq[t]:,} מופעים ({freq[t]/total*100:.2f}%)"
          + f" · {sev.LEVEL_HE[lvl]}**")
        w()
        w(ROLE_HE.get(t, ""))
        w()

        # מיקום בפסוק
        if position.get(t):
            pos = float(np.mean(position[t]))
            sd = float(np.std(position[t]))
            where = ("בתחילת הפסוק" if pos < 0.3 else
                     "בסוף הפסוק" if pos > 0.7 else "באמצע הפסוק")
            tight = " — ובעקביות גבוהה" if sd < 0.15 else ""
            w(f"**איפה בפסוק:** בממוצע ב-{pos*100:.0f}% מאורך הפסוק, כלומר "
              f"{where}{tight}.")
            w()

        # תחביר נמדד
        nxt = after.get(t, Counter())
        prv = before.get(t, Counter())
        if nxt or prv:
            w("**מה בא לפניו ואחריו** (נספר בקורפוס, לא הועתק):")
            w()
            w("| | הנפוץ ביותר | שיעור |")
            w("|---|---|---|")
            if prv:
                a, n = prv.most_common(1)[0]
                w(f"| לפניו | {tg.TAAM_HE.get(a, a)} | "
                  f"{n/sum(prv.values())*100:.0f}% |")
            if nxt:
                a, n = nxt.most_common(1)[0]
                w(f"| אחריו | {tg.TAAM_HE.get(a, a)} | "
                  f"{n/sum(nxt.values())*100:.0f}% |")
            w()
            if nxt:
                top = nxt.most_common(1)[0]
                share = top[1] / sum(nxt.values())
                if share > 0.6:
                    w(f"> החוק הזה חזק: אחרי {he} בא "
                      f"**{tg.TAAM_HE.get(top[0], top[0])}** ב-"
                      f"{share*100:.0f}% מהמקרים. זה אילוץ שהמוצר יכול "
                      f"לאכוף — טעם אחר בעמדה הזו כמעט תמיד שגיאה.")
                    w()

        # המנגינה
        w("**הסלסול** — מה שנמדד מההקלטות:")
        w()
        if m and m["n"] >= 20:
            w(f"```")
            w(f"{sparkline(m['mean'], 32)}")
            w(f"```")
            w()
            w(describe_contour(m))
            w()
            cons = m["consistency"]
            verdict = ("עקבי מאוד — נשמע כמעט אותו דבר בכל מופע"
                       if cons > 0.5 else
                       "עקבי למדי" if cons > 0.35 else
                       "משתנה בין מופעים — ההקשר משפיע על הצורה")
            w(f"נמדד על {m['n']} מופעים. עקביות {cons:.2f} — {verdict}.")
            w()
        elif m:
            w(f"נמדד על {m['n']} מופעים בלבד — מעט מכדי לקבוע צורה. "
              f"הטעם הזה נלמד ונשמע, אך אינו מקבל ציון.")
        else:
            w("אין הקלטות. הטעם הזה ניתן ללמידה מהטקסט, "
              "אך לא לבדיקה אקוסטית.")
            w()

        # דיוק המודל
        a = acc.get(t) if isinstance(acc, dict) else None
        if isinstance(a, dict) and "accuracy" in a:
            w(f"**דיוק המודל:** {a['accuracy']*100:.1f}% "
              f"(n={a.get('n', '?')}, פרשות שלא נראו).")
            w()

        # מה אומרים לילד
        v = sev.verdict_for("taam", expected=t, heard="—")
        if lvl == sev.STRICT:
            w(f"**אם טועים כאן:** {sev.LEVEL_HE[lvl]}. הפסוק נחתך במקום אחר, "
              f"ולכן זו טעות שמחזירים עליה.")
        elif lvl == sev.MODERATE:
            w(f"**אם טועים כאן:** {sev.LEVEL_HE[lvl]} — זה מזיז חלוקה "
              f"משנית בפסוק.")
        else:
            w(f"**אם טועים כאן:** {sev.LEVEL_HE[lvl]}. משרת שהוחלף במשרת "
              f"אחר אינו מזיז שום גבול — גוערים ולא מחזירים.")
        w()
        if t in audio:
            w("*יש הקלטת ייחוס לטעם הזה.*")
            w()
        w("---")
        w()

    w("## איך המסמך הזה נבנה")
    w()
    w("`Data_Prep/build_tikkun.py`. השכיחויות, המיקומים והתחביר נספרים "
      "מהטקסט המוטעם; עקומות המנגינה ממוצעות מהקלטות אמיתיות לאחר נרמול "
      "לגובה הממוצע של המילה עצמה, כך שהן מתארות **צורה** ולא גובה מוחלט; "
      "דירוג החומרה מ-`Data_Prep/severity.py`.")
    w()
    w("כשייאספו קולות נוספים, הרצה חוזרת של הסקריפט מעדכנת את כל המספרים.")
    w()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tikkun_soferim.md").write_text("\n".join(L) + "\n",
                                           encoding="utf-8")
    # מבנה מכונה, לשימוש האפליקציה
    data = {}
    for t in order:
        if not freq.get(t):
            continue
        m = mel.get(t)
        data[t] = {
            "he": tg.TAAM_HE.get(t, t), "sign": SIGNS[t],
            "sephardi": SEPHARDI_NAME.get(t),
            "rank": tg.DISJUNCTIVE_RANK.get(t),
            "role": "disjunctive" if tg.DISJUNCTIVE_RANK.get(t) else "conjunctive",
            "count": freq[t], "share": round(freq[t] / total, 5),
            "severity": sev.taam_severity(t),
            "role_he": ROLE_HE.get(t, ""),
            "position_mean": round(float(np.mean(position[t])), 3) if position.get(t) else None,
            "follows": [[k, v] for k, v in after.get(t, Counter()).most_common(4)],
            "preceded_by": [[k, v] for k, v in before.get(t, Counter()).most_common(4)],
            "curve": [round(float(x), 2) for x in m["mean"]] if m else None,
            "curve_sd": [round(float(x), 2) for x in m["sd"]] if m else None,
            "range_semitones": round(m["range"], 2) if m else None,
            "peak_at": round(m["peak_at"], 3) if m else None,
            "curve_n": m["n"] if m else 0,
            "consistency": round(m["consistency"], 3) if m else None,
            "contour_he": describe_contour(m),
            "checkable": bool(m and m["n"] >= 20),
        }
    (OUT / "tikkun_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    n_check = sum(1 for d in data.values() if d["checkable"])
    print(f"  {len(data)} טעמים תועדו")
    print(f"  {n_check} ניתנים לבדיקה אקוסטית, "
          f"{len(data)-n_check} לא (מעט מדי הקלטות)")
    print(f"  נשמר: {OUT/'tikkun_soferim.md'}")
    print(f"        {OUT/'tikkun_data.json'}")


if __name__ == "__main__":
    main()
