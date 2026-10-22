# AGENT_HANDOFF — Execute Taamify Research OS

> **משימתך:** לממש ולבחון את תוכנית המחקר והפיתוח של Taamify **מתוך ענף Taamify המקומי העדכני**, לא מתוך Dvir OS. המסמכים בתיקייה זו הם תוכנית עבודה עם ניסויים מדעיים והנדסת מוצר. אין לעשות push, merge, לפרוס, לשלוח אודיו, להתקין שירות חדש בתשלום או לשנות production checkpoint ללא אישור מפורש.

## סדר קריאה חובה

1. `docs/research-os/README.md`; `BRANCH_SYNC.md`; `ROADMAP.md`.
2. מקומית: `PLAN_STAGE_A.md`, `CLAUDE.md`, `run_queue.sh`, `build_ops_dashboard.py`, `webapp/server.py`, `Data_Prep/holdout.py`, `Data_Prep/calibrate_end_to_end.py`. אל תניח שהן קיימות בענף המרוחק.
3. `RESEARCH_PROTOCOL.md`, `PRODUCT_ACCEPTANCE.md`, `AUGMENTATION_PROTOCOL.md`, `TRAINING_HOURS.md`, `DVIR_OS_INTEGRATION.md`.
4. קונפיגורציות JSON תחת `research_os/config/`. סקריפט `research_os/plan.py` הוא כלי תכנון בלבד.

## Task 0 — מקור אמת ובדיקת הסביבה (חובה קודם)

- ודא מהו commit מקומי עדכני, אילו 53 קומיטים לא סונכרנו, האם משימות תור/שרת רצים והיכן נמצאים נתוני האמת. צור דוח `results/research_os/audit/local_audit.md` שיישאר מקומי אם כולל נתיבים/מידע מזהה.
- בצע בדיקת read-only: checkpoint שמוגש ב־API, תאימות verify-head, class labels (21?), נאמנות חלון, גרסאות manifest, בקשות API אמיתיות על קבוצת אימות.
- שחזר במידת האפשר את כל 8 תנאי שלב א׳, עם שמונת הפרשות, ממדדים שיצאו באמת מה־API, ואל תשתמש ב־random split או בסיווג ספקטרוגרמות שמורות בלבד.
- ודא ש־BEN13R metadata מצטרף דרך `verse+block` מדויק. **לעולם לא join לפי משך אודיו**; זו הייתה שגיאה מתועדת של 32%.
- ודא רישיונות, הסכמות/זכויות של האודיו, ושאין mixed-tradition leakage. עצור ודווח על אי התאמה.
- מדוד M1 Max GPU ורוחב זיכרון פנוי לפני פתיחת תורי אימון; אל תסתמך על '6 GPU slots' כששש עבודות MPS כבדות חולקות מאיץ יחיד.

**Definition of Done:** דוח מפורט, נתוני provenance, פערים, test outputs; ללא שינוי בפרודקשן.

## Task 1 — כיול ערוצי דקדוק, עבודה מוצרית ראשונה

- stress / dagesh / guttural / sheva: אחזר את 1,800 המדידות הקודמות במידת האפשר, בצע calibration על split ולידציה בנפרד, אחרי שער alignment ו־channel-specific windows.
- מדוד false accusation לכל ערוץ ולמוצר במצטבר, אמת recall של טעויות אמיתיות, coverage ואפשרות `לא בטוח` + uncertainty.
- השתמש רק בראיות אמת לטעויות אמיתיות. אל תייצר "טעות" על ידי בחירת label אחר לדוגמה נכונה.
- בדוק מה משתנה בשיעור false "צריך לחזור" על אודיו תקין; שמור threshold artifact עם hash.
- **לא** להחליף live threshold ללא סקירה.

**Definition of Done:** evaluation report, calibration candidate, paired before/after, live-API regression ללא side effects, הצעת PR נפרדת.

## Task 2 — Gold Dataset וקול שני

- הגדר מטריצה של פסוקים וטעמים, קוראים ספרדים־ירושלמים, גובה קול, טווח גיל, חדר/מיקרופון, ביצועי אמת נכונים/שגויים; זהות מקור מוקלט והסכמת שימוש.
- בצע איסוף קוראים אמיתיים לפי אישור הבעלים, לא scraping גורף; לייצר gold subset מסומן ידנית פעמיים + מדד הסכמה.
- שמור `speaker_id` בדוי ו־`source_root_id` קבוע לנגזרות; audio ו־PII נשארים רק מקומית. מבחן final של קוראים חדשים ייפתח **פעם אחת**.

**Definition of Done:** manifest versioned checksum, consent coverage report (ללא פרטים מזהים ב־Git), split leakage tests, ספקטרוגרמות מקור מסונכרנות ומאומתות.

## Task 3 — תשתית פייפליינים דטרמיניסטית

- חוזה train/val/test מסודר, seeds, hashes ל־dataset, feature, tokenizer/encoder, class list, model checkpoint, calibration ואומדן דקות לאפוק; תמיכה בשחזור וסגירת hooks.
- שמור מנוע התורים הקיים, לא להחליפו. היה מדוד על M1 Max לפני תכנון parallel GPU.
- בנה `BEN13R -> manifest` מדויק, `registry` של P00–P12, cache ל־frozen feature extractors, מאמני probe שנטענים ממניפסט ולא מנתיבים קשיחים.
- כל ריצה שומרת תוצאות על `train`/development בלבד עד פתיחת final test המאושרת; אין דוגמאות סינתטיות במבחן.
- הבדל ברור בין `planned`, `running`, `failed`, `passed_development`, `passed_locked_test`, `serving_production`.

**Definition of Done:** פיילוט P00/P01/P03, 3-seed reproducibility, logs ומדדים נקיים, בקרה נגד test leakage.

## Task 4 — Ablation ו־Augmentation מחקריים

- P01/P02/P06: מלודיה/משך/אנרגיה לעומת CNN קיים, חלון זמן זהה כקבוצת בקרה, והשוואה למודל פרודקשן.
- P03/P04: WavLM/W2v-BERT frozen embeddings (תחילה 2–3 epochs warmup), memory gate; P05 MuQ **מחקר לא־מסחרי בלבד**.
- P08 Voice Conversion: רק על קולות עם הסכמה מפורשת, תוך שמירת קונטור F0 ומשך/הברה וביקורת אנושית, מול Pitch Shift שכבר הוכיח עמידות לגובה קול. אין לפרש 10 גרסאות סינתטיות כ־10 דוברים.
- P07/P09/P10/P11/P12 רק כאשר התלויות זמינות (טעויות אמת, קוראים חדשים, מורה מאומת). אין להשתמש במדגם אחד לכל הטענות.
- כל ניסוי מוצהר מראש לפי H0–H7; negative controls כוללים shuffled labels, random same-width windows, held-out voices, מקור vs סינתטי ובדיקת תכונות דובר.

**Definition of Done:** paired metrics by reader and taam, false alarm budget, confidence intervals grouped by speaker, failure analysis including negative results.

## Task 5 — פרסום מדעי ומוצר במקביל

- כתוב `report/paper` עם שאלת מחקר, השערות, שיטות, שיקולי אתיקה, תוצאות מאומתות, ablation, שכפול ומגבלות. ללא הצגת ניחוש סטטיסטי כאימות.
- הכן מודל מועמד על בסיס Gates ב־`PRODUCT_ACCEPTANCE.md`, עם ONNX parity והשהיה מדודה, בדיקת API חי ושמירת גרסת prod קודם.
- אל תקדם מודל לפרודקשן אוטומטית; פתח PR עם בדיקות, ראיות ואישור בעלים.

## Task 6 — Dvir OS status export

- הבעלים רוצה תצוגת תורים, ניסויים, ניצול חומרה, hours, logs, hypothesis status ב־Dvir OS. פרמט `taamify-status.v1.json` מסונן **ללא אודיו/שמות/טקסט אישי**, עם SHA checkpoint והכרעת תוצאה.
- **אין** לאפשר Dvir OS לטעון מדעית שנמדד קול חדש כשמדובר בקול אחד. אין לזרום רשומות אודיו דרך Drive או GitHub.
- לאחר אישור אבטחה, ניתן לממש adapter לקריאה בלבד דרך חיבור מקומי מאומת; סטטוס ללא bridge מצוין כ־`not_connected`, ולא "running".

**Definition of Done:** סכמת JSON, בדיקות sanitization, חיבור בהמתנה לאישור הבעלים, UI מפריד תכנון מהוכחות.

## כללי מדידה ותקציב

- זמן מאיץ **אינו** שעות מחקר אקדמי של אדם. שעות אקדמיות מתועדות לפי עבודה וממצא; `TRAINING_HOURS.md` מפריד טווחי אדם מ־fit-epochs.
- שעות מאיץ מחושבות **ממדידות אמיתיות** לכל family/host. אל תאמץ אומדני 156h משיחה קודמת כאילו נמדדו.
- הדו"ח צריך להיות שימושי גם למחקר מדעי וגם למוצר, כולל ניסויים שנכשלו.

## הודעת פתיחה מומלצת לסוכן

> קרא את `docs/research-os/AGENT_HANDOFF.md` ואת `BRANCH_SYNC.md`. ודא שבידך ה־checkout המקומי עם 53 הקומיטים שלא נדחפו. אל תדרוס עבודה מקומית, אל תפרוס ואל תפעיל jobs לפני השלמת S0. תן תחילה דוח Source-of-Truth Audit, אחר כך בצע כיול ערוצים כשינוי נפרד, ואז התחל ניסויי המחקר לפי החסימות והפרוטוקול. כל קוד ישאר ב־Taamify; Dvir OS מקבל רק מטא־נתונים ואישורי סקירה.
