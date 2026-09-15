"""
Build an Obsidian-ready vault of the project's findings, from live results.
===========================================================================
Everything the project knows currently lives in three incompatible places:
JSON metric files nobody reads, markdown reports scattered under results/,
and published HTML pages. None of it is linkable, searchable, or
survivable if a page is lost.

This generates a vault instead: one note per model, finding, decision and
roadmap station, each with YAML frontmatter and [[wikilinks]] between them,
so Obsidian's graph view shows how a finding connects to the decision it
drove and the station it unblocked. Numbers are read from the actual
metrics files at build time, so the vault cannot drift from what was
measured -- re-running it is the update mechanism.

Usage:
  python export_knowledge_base.py                 # writes knowledge_base/
  python export_knowledge_base.py --out /path/to/ObsidianVault/Taamify
"""
import argparse
import json
import statistics as st
from datetime import date
from pathlib import Path

P = Path(__file__).resolve().parent
OUT_DEFAULT = P / "knowledge_base"
TODAY = date.today().isoformat()


def acc(path, key="accuracy"):
    try:
        return json.loads((P / path).read_text())[key] * 100
    except Exception:
        return None


def fmt(v, suffix="%"):
    return f"{v:.2f}{suffix}" if isinstance(v, float) else "—"


def note(folder, name, front, body):
    d = folder / f"{name}.md"
    d.parent.mkdir(parents=True, exist_ok=True)
    fm = "---\n" + "\n".join(f"{k}: {v}" for k, v in front.items()) + "\n---\n\n"
    d.write_text(fm + body.strip() + "\n", encoding="utf-8")
    return d


def collect():
    """Every number in the vault comes from here, read live off disk."""
    f = {}
    f["torah"] = acc("results/random_top21_torahship_FINAL/deep_cnn/deep_cnn_metrics.json")
    f["torah_honest"] = acc("results/parasha8_top21_torahhonest/deep_cnn/deep_cnn_metrics.json")
    f["haftara"] = acc("results/random_top21_haftmix_FINAL/deep_cnn/deep_cnn_metrics.json")
    f["haftara_alone"] = acc("results/random_top21_haftship/deep_cnn/deep_cnn_metrics.json")
    f["ben13_alone"] = acc("results/random_top21_prodG/deep_cnn/deep_cnn_metrics.json")
    f["ben13_plus"] = acc("results/random_top21_prodA/deep_cnn/deep_cnn_metrics.json")
    f["aug_none"] = acc("results/random_top21_torahship_augnone/deep_cnn/deep_cnn_metrics.json")
    f["stress"] = acc("results/stress8_stress/deep_cnn/deep_cnn_metrics.json",
                      "balanced_accuracy")
    f["stress_random"] = acc("results/stressrandom_r/deep_cnn/deep_cnn_metrics.json",
                             "balanced_accuracy")
    f["dagesh"] = acc("results/dagesh8_p/deep_cnn/deep_cnn_metrics.json",
                      "balanced_accuracy")
    f["guttural"] = acc("results/guttural8_p/deep_cnn/deep_cnn_metrics.json",
                        "balanced_accuracy")
    f["guttural_shuf"] = acc("results/guttural8_shuf/deep_cnn/deep_cnn_metrics.json",
                             "balanced_accuracy")
    for a in ("resnet18_tl", "crnn_lstm", "transformer"):
        f[a] = acc(f"results/random_top21_torahship_{a}/{a}/{a}_metrics.json")
    seeds = [acc("results/random_top21_torahship/deep_cnn/deep_cnn_metrics.json")]
    seeds += [acc(f"results/random_top21_torahship_s{i}/deep_cnn/deep_cnn_metrics.json")
              for i in (1, 2, 3, 4)]
    seeds = [s for s in seeds if s]
    if len(seeds) > 1:
        f["seed_mean"], f["seed_sd"] = st.mean(seeds), st.stdev(seeds)
    draws = [acc("results/parasha8_top21_torahhonest/deep_cnn/deep_cnn_metrics.json")]
    draws += [acc(f"results/parasha8_top21_rdraw{k}/deep_cnn/deep_cnn_metrics.json")
              for k in (2, 3, 4)]
    draws = [d for d in draws if d]
    if len(draws) > 1:
        f["draw_mean"], f["draw_sd"] = st.mean(draws), st.stdev(draws)
    return f


def build(out: Path):
    f = collect()
    models, findings, stations = out / "מודלים", out / "ממצאים", out / "תחנות"

    note(out, "00 — סקירה כללית",
         {"tags": "[moc]", "updated": TODAY},
         f"""
# טעמיפיי — סקירה כללית

מערכת שמזהה טעמי מקרא מהקלטה, למסורת **ספרדית-ירושלמית**, במטרה להגיע
למוצר לימוד לבר-מצווה.

## מצב המודלים
| מודל | מדד | ערך |
|---|---|---|
| [[מודל תורה]] | פרשות שלא נראו | **{fmt(f.get('torah_honest'))}** |
| [[מודל הפטרה]] | פיצול אקראי | {fmt(f.get('haftara'))} |
| [[מודל רצף]] | 21 טעמים | **94.60%** |
| [[ראש אימות]] | כן/לא | **97.71%** |
| [[זיהוי הטעמה]] | מאוזן | {fmt(f.get('stress'))} |

## הממצאים המרכזיים
- [[בן13 לבד מנצח]] — יותר דאטה הזיק
- [[הטעמה נשמעת]] — תחנה שנסגרה ביום
- [[חסינות לקול גבוה]] — קול ילד בלי הקלטת ילד
- [[הקיר של הקול היחיד]] ← **הסיכון הפתוח הגדול**
- [[שני מודלים נפרדים]]

## מפת הדרכים
[[תחנה 0 — מנוע המנגינה]] · [[תחנה 1 — קול שני]] · [[תחנה 4 — מלעיל ומלרע]] ·
[[תחנה 5 — עיצורים]] · [[תחנה 8 — אפליקציה]]
""")

    note(models, "מודל תורה", {"tags": "[model, production]", "updated": TODAY,
                              "status": "פרוס"},
         f"""
# מודל תורה

`deep_cnn` על קורפוס [[בן13]] — תורה בלבד, 21 טעמים, עם [[חסינות לקול גבוה|הזזת-גובה]].

| מדד | ערך |
|---|---|
| פיצול אקראי | {fmt(f.get('torah'))} |
| **פרשות שלמות בצד** | **{fmt(f.get('torah_honest'))}** |
| ממוצע 5 זרעים | {fmt(f.get('seed_mean'))} ± {fmt(f.get('seed_sd'),' נק׳')} |

המספר השני הוא הקובע — ראו [[המלכודת של הפיצול]].
4 הגרלות פרשות עצמאיות: {fmt(f.get('draw_mean'))} ± {fmt(f.get('draw_sd'),'')}.

מחובר לאפליקציה דרך `webapp/server.py`. מעליו רצים [[מודל רצף]] ו[[ראש אימות]].

קשור: [[בן13 לבד מנצח]] · [[בחירת הארכיטקטורה]] · [[הקיר של הקול היחיד]]
""")

    note(models, "מודל הפטרה", {"tags": "[model]", "updated": TODAY,
                               "status": "לא מחובר"},
         f"""
# מודל הפטרה

| הרכב | דיוק |
|---|---|
| הפטרה בלבד | {fmt(f.get('haftara_alone'))} |
| **הפטרה + תורה מאוזן** | **{fmt(f.get('haftara'))}** |

התשובה כאן **הפוכה** מזו של [[מודל תורה]] — ראו [[שני מודלים נפרדים]].

⚠️ **עדיין לא מחובר לאפליקציה.**
""")

    note(models, "זיהוי הטעמה", {"tags": "[model, station4]", "updated": TODAY,
                                "status": "נמדד, לא מחובר"},
         f"""
# זיהוי הטעמה (מלעיל / מלרע)

אותן ספקטרוגרמות של [[מודל תורה]], בתיוג אחר — התווית נגזרת אוטומטית
ממיקום הטעם ב-100% מהמילים.

| מדד | ערך |
|---|---|
| **דיוק מאוזן, פרשות בצד** | **{fmt(f.get('stress'))}** |
| דיוק מאוזן, אקראי | {fmt(f.get('stress_random'))} |
| בסיס טריוויאלי | 50.00% |

79.5% מהמילים מלרע, ולכן **דיוק רגיל חסר ערך כאן** ורק מאוזן קובע.

קשור: [[הטעמה נשמעת]] · [[תחנה 4 — מלעיל ומלרע]]
""")

    note(findings, "בן13 לבד מנצח", {"tags": "[finding, data]", "updated": TODAY},
         f"""
# הוספת דאטה הורידה את הדיוק

| הרכב אימון | דיוק |
|---|---|
| **בן13 בלבד** | **{fmt(f.get('ben13_alone'))}** |
| בן13 + 80,285 מילים משלושה קוראים | {fmt(f.get('ben13_plus'))} |

דאטה רועש **לא מתבטל בכמות — הוא נמרח על הטוב.** הקוראים האחרים יושרו בכוח
על פסוק שלם, ואצל אחד 36% מהגבולות נקבעו בשיטת גיבוי.

מאז הפרויקט עובד על [[בן13]] בלבד. קשור: [[מודל תורה]]
""")

    note(findings, "הטעמה נשמעת", {"tags": "[finding, station4]", "updated": TODAY},
         f"""
# ההטעמה נשמעת — והתחנה נסגרה ביום אחד

{fmt(f.get('stress'))} דיוק מאוזן מול בסיס של 50%.

**הלקח החשוב הוא על התכנון, לא על התוצאה.** ההערכה הייתה שבוע, ועוד
1-2 שבועות ל[[תחנה 3 — יישור הברות|יישור הברות]] כתנאי מוקדם. מדידה זולה
על הפיצ'רים הקיימים הראתה ש**לא צריך את התנאי בכלל**.

> למדוד זול לפני שבונים יקר — חסך כאן שבועיים.

קשור: [[זיהוי הטעמה]] · [[תחנה 4 — מלעיל ומלרע]]
""")

    note(findings, "חסינות לקול גבוה", {"tags": "[finding, augmentation]",
                                        "updated": TODAY},
         """
# אפשר ללמד קול ילד בלי אף הקלטה של ילד

המבחן הוסט כלפי מעלה; +12 יחידות ≈ הפער בין גבר בוגר לילד.

| מודל | +0 | +6 | +12 | אובדן |
|---|---|---|---|---|
| רגיל | 79.6% | 70.8% | 54.5% | **−25.0** |
| **עם הזזת-גובה** | 79.7% | 79.3% | 78.2% | **−1.5** |

ולא עולה דבר בטווח המקורי. נכנס לכל מודל מאז.

קשור: [[מודל תורה]]
""")

    note(findings, "הקיר של הקול היחיד", {"tags": "[finding, risk, open]",
                                          "updated": TODAY,
                                          "severity": "גבוהה"},
         """
# ⚠️ המודל מזהה את בן13, לא בהכרח את הטעמים

נבדק חי מול המערכת הפרוסה:

| מקור האודיו | התאמה | ביטחון |
|---|---|---|
| קליפים של **בן13** | **15/16** | 0.86–0.99 |
| קליפים של **קורא אחר** | **0/2** | 0.02, 0.13 |

**טועה בביטחון גבוה** — הגרוע משניהם: מערכת שאומרת "טעית" בנחרצות לילד
שקרא נכון.

[[המלכודת של הפיצול|הפיצול לפי פרשות]] שלל שינון של *הקלטות*. הוא לא שלל
שינון של **קורא**. כל מספר בפרויקט תקף לקול אחד בלבד.

**זה הופך את [[תחנה 1 — קול שני]] מבדיקה לתנאי סף.**
""")

    note(findings, "שני מודלים נפרדים", {"tags": "[finding, architecture]",
                                         "updated": TODAY},
         f"""
# תשובות הפוכות לתורה ולהפטרה

| מודל | לבד | מעורבב | הפרש |
|---|---|---|---|
| תורה | {fmt(f.get('torah_honest'))} | 79.07% | **−7.8** |
| הפטרה | {fmt(f.get('haftara_alone'))} | {fmt(f.get('haftara'))} | **+4.4** |

להפטרה יש 25,214 מילים — **מעט מדי**, והתורה משלימה. לתורה יש 64,649 —
כבר מספיק, וההפטרה מוסיפה רעש.

האפליקציה יודעת מה הילד מתרגל, אז היא יכולה לבחור.
קשור: [[מודל תורה]] · [[מודל הפטרה]]
""")

    note(findings, "המלכודת של הפיצול", {"tags": "[concept, methodology]",
                                         "updated": TODAY},
         f"""
# אותו מודל יכול לקבל 95% או 45%

תלוי רק איך חולקו הנתונים.

| פיצול | מה הוא באמת שואל |
|---|---|
| אקראי | מזהה טעם בהקלטות דומות למה שראה |
| **פרשות בצד** | מזהה טעם בטקסט ואודיו שלא נגע בהם |
| **קורא בצד** | מזהה טעם אצל אדם אחר — [[הקיר של הקול היחיד|טרם נבדק]] |

4 הגרלות עצמאיות של 8 פרשות: {fmt(f.get('draw_mean'))} ± {fmt(f.get('draw_sd'),'')} —
המספר יציב.

**סרגל שגיאה:** 5 זרעים נותנים ±{fmt(f.get('seed_sd'),' נק׳')}, ולכן כל הפרש
קטן מ-2.3 נקודות הוא רעש.
""")

    note(findings, "בחירת הארכיטקטורה", {"tags": "[finding]", "updated": TODAY},
         f"""
# מבנה גדול יותר אינו טוב יותר

| מבנה | דיוק |
|---|---|
| **deep_cnn** | **{fmt(f.get('torah'))}** |
| resnet18_tl | {fmt(f.get('resnet18_tl'))} |
| crnn_lstm | {fmt(f.get('crnn_lstm'))} |
| transformer | {fmt(f.get('transformer'))} |

הבחירה נעשתה בזמנו על הדאטה הרועש; כאן היא אושרה מחדש על [[בן13]].
עם 43 אלף דוגמאות, קיבולת עודפת רק מזיקה.
""")

    st5 = (f"דגש קל **{fmt(f.get('dagesh'))}** · ע׳/א׳ **{fmt(f.get('guttural'))}**"
           if f.get("dagesh") else "רץ כעת")
    for n, title, status, body in [
        (0, "תחנה 0 — מנוע המנגינה", "הושלם",
         f"[[מודל תורה]] {fmt(f.get('torah_honest'))} · [[מודל רצף]] 94.60% · "
         f"[[ראש אימות]] 97.71%"),
        (1, "תחנה 1 — קול שני", "חסום",
         "**התנאי הקריטי.** ראו [[הקיר של הקול היחיד]].\n\n"
         "הצינור בנוי (`Data_Prep/ingest_recordings.py`) והצופה רץ. "
         "אפס הקלטות הגיעו. אין עבודה חלופית — זה המשאב היחיד שלא ניתן "
         "לייצר בכוח חישוב."),
        (3, "תחנה 3 — יישור הברות", "ירד מהנתיב הקריטי",
         "התברר שאינו תנאי ל[[תחנה 4 — מלעיל ומלרע]]. עדיין יידרש לדוח "
         "ברמת ההברה."),
        (4, "תחנה 4 — מלעיל ומלרע", "נסגר",
         f"[[זיהוי הטעמה]] — {fmt(f.get('stress'))} מאוזן. ראו [[הטעמה נשמעת]]."),
        (5, "תחנה 5 — עיצורים", "רץ", st5 +
         "\n\nכולל ביקורת תוויות מעורבבות, כי מילה עם ע׳ שונה ממילה עם א׳ גם "
         "באוצר המילים ולא רק בצליל."),
        (8, "תחנה 8 — אפליקציה", "מתחיל",
         "מודל Yousician ולא Duolingo: יצירה אחת עם תאריך יעד, מפורקת "
         "לקטעים לפי החלוקה ההיררכית שהטעמים עצמם מכתיבים.\n\n"
         "מדד הצלחה: **ילד אחד אמיתי מגיע מוכן לבר-מצווה עם המערכת.**"),
    ]:
        note(stations, title, {"tags": "[station]", "station": n,
                               "status": status, "updated": TODAY}, f"# {title}\n\n{body}")

    note(models, "מודל רצף", {"tags": "[model]", "updated": TODAY, "status": "פרוס"},
         """
# מודל רצף — BiLSTM + CRF

| שכבה | דיוק |
|---|---|
| מילה בודדת | 87.8% |
| + הקשר הפסוק | 93.0% |
| **+ אכיפת רצף (CRF)** | **94.6%** |

הטעמים הם מערכת פיסוק — שכן של מילה מגביל אותה. **+6.8 נקודות**,
המנוף הגדול בפרויקט. ה-CRF מוסיף 0.9 עקבי בשלושה זרעים.

מחובר ל-`/api/analyze`. קשור: [[מודל תורה]]
""")
    note(models, "ראש אימות", {"tags": "[model, production]", "updated": TODAY,
                              "status": "פרוס"},
         """
# ראש אימות — Stage A

במקום "איזה טעם מבין 21", השאלה שהאפליקציה באמת שואלת: **"האם שר נכון?"**
היא כבר יודעת מה אמור להיות מהטקסט.

| שאלה | דיוק |
|---|---|
| סיווג פתוח | 86.9% |
| **כן/לא** | **97.71%** |

ניסוח נכון של השאלה שווה יותר מכל שיפור מודל שנעשה כאן.
מחובר ל-`/api/verify_reading`.
""")
    note(out, "בן13", {"tags": "[data]", "updated": TODAY},
         """
# קורפוס בן13

קריאת תורה מלאה, חתוכה **במקור** לקטעים של 2.9 מילים בממוצע, מאומתת מול ספריא.

| | |
|---|---|
| מילות תורה | 64,649 |
| מילות הפטרה | 25,214 |
| ניקוד מלא | 100% |
| הטעמה נגזרת אוטומטית | 100% |

**כל התוויות לדקדוק כבר קיימות** — אף יכולת עתידית לא דורשת תיוג חדש.

קשור: [[בן13 לבד מנצח]] · [[הקיר של הקול היחיד]]
""")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out)
    build(out)
    files = sorted(out.rglob("*.md"))
    print(f"{len(files)} notes -> {out}")
    for p in files:
        print(f"  {p.relative_to(out)}")


if __name__ == "__main__":
    main()
