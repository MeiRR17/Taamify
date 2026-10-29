# T04 — עבודה קודמת ישירה: הטכניון, Cantillate, TaamimFlow, PocketTorah

**תאריך בדיקה:** 29.10.2026 · **סטטוס:** `verified_current` לסעיפים שסומנו "נמדד"; השאר קריאת קוד/README.
**כלים:** `research_os/technion_split_overlap.py`, `research_os/build_diagnostic_pack.py --benchmark-blocks`,
`research_os/benchmark_candidates.py` (ענף `codex/channel-calibration-review-20261027`, commit `687a3f7`).

## השורה התחתונה

1. **מודל הטכניון לנוסח ירושלמי אומן על בן13 — אותו קורא, אותן הקלטות כמו שלנו.** לכן כל השוואה על בן13
   חייבת להוציא את מה שהוא ראה באימון. ההשוואה מ־28.10 (99.0% מול 89.4%) לא עשתה זאת:
   37 מתוך 40 המשפטים היו בסט האימון שלו.
2. **על 160 משפטים שאף אחת מהמערכות לא אימנה עליהם, הוא עדיין טוב יותר בזיהוי:** 91.3% מול 84.3%,
   הפרש מזווג +6.8 נקודות, רווח סמך 95% [+2.4, +11.4]. גבוה מרף הרעש (~2.3), אבל הגבול התחתון קרוב אליו.
3. **על קוראים אחרים המודלים שלהם קורסים.** בקובץ המדידות שלהם על "קוראים שלא באימון": F1 מדויק
   0.05–0.52, ולמודל הירושלמי 0.26 עם WER 79%. אף מערכת — שלהם או שלנו — לא הוכיחה הכללה לקורא חדש.
4. **לאף אחד מארבעת הריפוזיטוריז אין רישיון קוד.** אסור להעתיק קוד. המשקולות ב־HF מוצהרות Apache-2.0,
   אבל אומנו על בן13 שזכויותיו לא הוסדרו (D01).

המשמעות: החידוש המדעי של Taamify **אינו** "זיהוי טעמים מאודיו" — זה קיים. החידוש האפשרי הוא
הכללה לקוראים חדשים באותו נוסח, הבחנה בין טעם שבוצע לטעם הכתוב, ומשוב טעויות מכויל (False accusation
מבוקר). הכול עדיין תלוי בקוראים אמיתיים נוספים (G2).

## 1. staviv/cantillation — SIPL, הטכניון

| שדה | ממצא | ראיה |
|---|---|---|
| מה זה | Whisper מכוונן לתמלול פסוק עם ניקוד וטעמים; בוט טלגרם | README, `main.py` |
| HEAD | `fc48793`, 07.05.2025, 89 commits | `git log` |
| רישיון קוד | **אין** — אף קובץ LICENSE בכל ההיסטוריה; GitHub API: `license=null` | `git log --diff-filter=A -- '*LICENSE*'` = 0 |
| נתונים | בן13 בארבעה נוסחים (maroko, yerushalmi, ashkenazi, bavly) + PocketTorah | README, `03_dataset.json` |
| הפיצול של מודל 24.07.2024 | 80/10/10 ברצפים של 50 שורות, `seed=42`; השורה = בלוק בן13 על כל ארבעת הנוסחים | `split_data.py@6f16c5f`, `jsons/*_data.json@6f16c5f` |
| תקלה בפיצול המאוחר | ב־19.03.2025 שונה ל־0.9/0.1 ⇒ `validation_test_ratio=1.0` ⇒ **סט test ריק** | `split_data.py@fc48793` |
| הערכה על קוראים אחרים | F1 מדויק 0.045–0.517; ירושלמי: 0.260, WER 79.1% | `evalutions_on_other_data/test_results.csv@84d2c86` |
| מדדים | Exact / Letter_Shift / Word_Level / Word_Shift — סובלניים למיקום, לא macro-F1 | `cantilLocations_evaluation.py` |
| משקולות | `cantillation/Teamim-small_..._nusach-yerushalmi_date-24-07-2024` rev `53028d4`, Apache-2.0 לפי הכרטיס | HF API (נבדק 27.10) |

### 1א. נמדד: כמה מבן13 שלנו נמצא באימון שלהם

ה־join מדויק לפי שם קובץ בן13 (`yerushalmi_{parasha}-{name}_{aliya}_{b}{cc}{vv}_{block}.mp3`), אותו מבנה
URL כמו ב־`data/ben13_reading/manifest_yerushalmi.json`. 40/40 משפטים עברו גם בדיקת התאמת טקסט.
אין שורה ששייכת לשני פיצולים. בין 17.07 ל־05.08.2024 אף commit לא שינה את `jsons/`.

| 8 הפרשות המוחזקות שלנו (5, 6, 8, 13, 29, 37, 41, 42) | בלוקים |
|---|---:|
| באימון הטכניון | 3,511 (79%) |
| ב־validation שלהם (שימש לבחירת checkpoint) | 467 |
| ב־test שלהם (לא שימש) | 448 |
| חסר בפיצול שלהם | 2 |

| ה־benchmark מ־28.10 (40 משפטים) | משפטים | טעמים | הטכניון | ה־API שלנו |
|---|---:|---:|---:|---:|
| באימון הטכניון | 37 | 97 | **100%** | 88.7% |
| מחוץ לאימון | 3 | 7 | 85.7% | 100% |

### 1ב. נמדד: השוואה הוגנת על בלוקים שאף מערכת לא ראתה

160 משפטים רציפים ולא חתוכים, 20 מכל פרשה, מתוך 902 הבלוקים ברשימת ההיתר
(allowlist SHA-256 `92423cb5…`). 446 טעמים. שרת בדיקה פרטי על 8078 בתצורת ברירת המחדל; המודל
החיצוני על MPS, מקומי, עם revision קבוע. האודיו לא יצא מהמחשב.

| מדד | הטכניון (Whisper-small ירושלמי) | ה־API שלנו |
|---|---:|---:|
| זיהוי: הטעם שנשמע = הטעם הכתוב | **91.3%** [88.3, 94.0] | **84.3%** [80.7, 87.8] |
| הפרש מזווג (bootstrap על משפטים, 10,000) | +6.8 [+2.4, +11.4] | |
| על ה־test שלהם בלבד (86 משפטים, 240 טעמים) | 92.5% | 85.0% |
| על ה־validation שלהם (74 משפטים, 206 טעמים) | 89.8% | 83.5% |
| פרשות שבהן עדיף | 7 מתוך 8 | 1 (פרשה 8: 94.8 מול 86.2) |
| p50 זמן לבקשה | 0.29 ש׳ | 0.25 ש׳ |

**הוגנות ההשוואה:** ה־API מקבל את המילים הצפויות ליישור; המודל החיצוני מפענח בלי טקסט וחייב לקבל
נכון גם את המילה. היתרון המבני הוא שלנו, ובכל זאת הוא מנצח. תוצאות פרוסות לפי פיצול/פרשה הן אבחון,
לא ממצא נפרד.

**מה זה לא:** לא הכללה לקורא חדש (שתי המערכות אומנו על הקול הזה). לא גילוי טעויות לומד. התוויות
הן הטעם הכתוב, לא ביצוע מתויג אנושית. מדגם יחיד, seed בחירה `20261028`.

### 1ג. תצפית לוואי — לבירור, לא ממצא

באותו מדגם, `/api/check_phrase` סימן **"שגוי"** ל־18 מתוך 446 טעמים בקריאה תקינה: **4.0%** [2.6, 6.3]
(Wilson). A3 של שלב א׳ (2.1%) נמדד במסלול אחר; F10 בתיק הראיות מציין שהמסלולים אינם אותו חישוב.
דרוש T01: ליישב את הגדרת A3 עם המסלול שהמסך בפועל קורא לו.

## 2. kylemath/cantillate

| שדה | ממצא |
|---|---|
| מה זה | אפליקציית ווב ללימוד פרשה/הפטרה: מצב מודרך לפי תאריך, השוואת קו גובה (pitch) חי מול ההקלטה, ציון התאמה 0–100, heatmap התקדמות |
| HEAD | `1129114`, פעיל (03.10.2026), 59 commits |
| נוסח | **אשכנזי** בלבד; מוטיבים "מקורבים" + צורות שנמדדו מהקלטות PocketTorah |
| זיהוי טעמים | **אין מסווג.** משווה עקומות גובה; לא קובע איזה טעם בוצע |
| רישיון קוד | **אין** (`license=null`); תוכן: PocketTorah CC-BY-SA, MAM CC-BY, גופנים GPL+FE/OFL |
| לקחת ממנו | UX בלבד: אשף לפי תאריך השמחה → פרשה → תוכנית; סבבי תרגול; "למה נבחרה המשימה". **לא** קוד ולא נכסים |

## 3. Moriahise/taamimflow_project

| שדה | ממצא |
|---|---|
| מה זה | שחזור "clean-room" של TropeTrainer ב־PyQt6: תצוגה, לוח קריאות, סינתזה (עדיין stub) |
| HEAD | `828854b`, 19.02.2026, 22 commits |
| זיהוי/משוב | **אין** |
| רישיון | **אין**; בנוסף הריפו כולל קבצים מהמתקין המקורי (`d2htls32.dll`, `tropedef.xml`, `*.ttt`) — סיכון משפטי |
| החלטה | **נדחה** לשימוש. רלוונטי רק כרקע שוק |

## 4. rneiss/PocketTorah

`license=null` ב־GitHub; ההקלטות מופצות CC-BY-SA לפי Cantillate. נוסח אשכנזי ⇒ **אסור כדאטה לאימון
Taamify** (CLAUDE.md: נוסח אחר = תווית שגויה). עדכון אחרון 2022.

## 5. טבלת חידוש מדעי

| טענה | הטכניון | Cantillate | TaamimFlow | Taamify — מצב |
|---|---|---|---|---|
| זיהוי טעם מאודיו | ✔ (Whisper, תמלול) | ✘ | ✘ | ✔ (CNN למילה); **חלש מהטכניון** על אותו קורא |
| נוסח ספרדי־ירושלמי | ✔ (אחד מארבעה) | ✘ | חלקי (סינתזה) | ✔ ייעודי |
| הערכה על קורא שלא באימון | ✔ ומראה קריסה | ✘ | ✘ | ✘ — **הפער המרכזי, חסום ב־G2** |
| הבחנת "בוצע" מול "כתוב" | ✘ (מנבא כתוב) | ✘ | ✘ | חלקי: חוזה intake קיים, אין תוויות ביצוע |
| משוב טעות מכויל (false accusation) | ✘ | ✘ | ✘ | ✔ מסגרת קיימת; A3 לא עובר במפורש |
| טעויות לומד אמיתיות | ✘ | ✘ | ✘ | ✘ — פרוקסי בלבד |
| פיצול נקי מזליגה | ✘ (ברצפים; test ריק בגרסה מאוחרת) | — | — | ✔ (פרשות, ופיצול עצמאי 28.10) |
| רישיון קוד | אין | אין | אין | — |

## 6. החלטות והמשך

- **לא** לטעון עליונות של מסווג Taamify על פני הטכניון. ההפך נמדד, על אותו קורא.
- **לבחון את משקולות הטכניון כבסיס/מקודד** (מועמד חדש ל־G3, למשל encoder קפוא + ראש למילה, או
  כ־challenger מלא). חסום חלקית ב־D01 (זכויות בן13) לשימוש מסחרי; מחקרית מותר מקומית.
- **השאלה המכריעה נשארת:** מה קורה שתי המערכות על קורא ירושלמי שני. בלי G2 אין תשובה.
- T01: ליישב 4.0% "שגוי" ב־`check_phrase` מול A3.
- להוסיף לכל benchmark עתידי מול מודל חיצוני את סינון החפיפה כברירת מחדל.

## שחזור

```bash
git clone https://github.com/staviv/cantillation /tmp/x/cantillation
mkdir split && for s in train validation test; do
  git -C /tmp/x/cantillation show 6f16c5f:jsons/${s}_data.json > split/${s}_data.json; done
venv/bin/python -m research_os.technion_split_overlap --split-dir split \
  --out results/research_os/benchmarks/technion_overlap_NEW.json \
  --allowlist results/research_os/benchmarks/technion_heldout_blocks_NEW.json \
  --benchmark-manifest results/research_os/diagnostic_pack_20261028/benchmark_manifest.jsonl \
  --benchmark-observations results/research_os/benchmarks/jerusalem_small_20261028/observations.json
venv/bin/python -m research_os.build_diagnostic_pack --max-words 1 --per-parasha 20 \
  --benchmark-blocks results/research_os/benchmarks/technion_heldout_blocks_NEW.json \
  --out results/research_os/diagnostic_pack_NEW
# שרת בדיקה פרטי על 8078 (לא 8000), ואז:
venv/bin/python -m research_os.benchmark_candidates --api http://127.0.0.1:8078 \
  --external yerushalmi-small --manifest results/research_os/diagnostic_pack_NEW/benchmark_manifest.jsonl \
  --out results/research_os/benchmarks/NEW
```

תוצרים מקומיים (לא ב־Git, כוללים אודיו/נתוני בן13): `results/research_os/benchmarks/technion_overlap_20261029.json`,
`technion_heldout_blocks_20261029.json`, `technion_heldout_20261029_r2/`, `diagnostic_pack_technion_heldout_20261029/`.
SHA-256 של קובצי הפיצול: train `f499120a…`, validation `2fea104c…`, test `f6551f7a…`.
