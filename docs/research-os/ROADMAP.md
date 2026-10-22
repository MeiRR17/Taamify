# Taamify — Roadmap תוכנית העבודה המשותפת למחקר ולמוצר

## סדר שלבים; אין להתחיל downstream לפני שהשער הקודם עבר

### S0 — Reconcile + Reproduce (עדיפות 0)
1. סטטוס Git: branch, SHA, פער 53 קומיטים, audit קבצים פרטיים ושימור תוצרים מקומיים; בלי push/merge ללא אישור.
2. לקרוא `PLAN_STAGE_A.md` המעודכן ואת הקוד שנשלח לשרת; לבדוק שבעיית התאמת בלוקים שנפתרה נשמרה.
3. לשחזר את שמונת מדדי שלב א' בהקלטות מ־8 הפרשות דרך ה־API עצמו. להפריד מודל טעם מערוצי דקדוק, לבדוק active checkpoint/verify head, split ו־labels.
4. לאשר מקור אמת לאודיו, לפרוטוקול, ולמחשב M1 Max + מרחק מה־M4/Windows.
5. לכתוב baseline card עם provenance; אם מדדים אינם משתחזרים — **עצור**, וחקור באג במקום לפתוח ניסוי AI יקר.

**Outputs:** `results/research_os/baseline_reproduction.json`, `audit.md` (מקומיים בלבד אם מכילים מידע פרטי), פערים עם עדיפות, מסמך הפרדת 8 הפרשות לעומת קוראים חדשים.

### S1 — Product calibration + feasibility research (מותר באותו זמן לאחר S0)
1. כיול channels: stress, dagesh, guttural, sheva; ספים פר־ערוץ ו־abstention; מדידת false accusation ו־recall בדאטה נפרד.
2. לבדוק P00/P01/P02/P03/P06 עם controlled seed ראשון, ממשק features ו־train/eval; memory/profiling.
3. P04 W2v-BERT ו־P05 MuQ רק אחרי gates RAM ו־license; לא להפעיל אוטומטית.
4. עדכן זמני epoch אמיתיים, feature extraction, storage I/O. one-heavy-job-per-MPS device כנקודת התחלה.

**Outputs:** מסמכי calibration מתועדים + 1-seed pilot candidates + utilization/epoch logs. אין טענת cross-speaker transfer.

### S2 — Gold set וגיוס דוברים בלתי תלויים (מסלול מקביל ל־S1)
1. תכנון reader diversity / controlled verse inventory — אותו נוסח ספרדי־ירושלמי.
2. גיוס מורשה של קורא שני לפחות; יעד תכנון של עד 8 קוראים, בכפוף למבחן power.
3. תיוג acoustic אמת (taam intended/performed) ואיכות alignment; דגימות "קורא נכון" ו"קורא טועה" שאינן תוצר של ניחוש label של מודל.
4. freeze final test מראש כולל כל נגזרות source_root.

**Outputs:** gold manifest versioned, data card, agreement report, held-out speaker protocol.

### S3 — Confirmatory candidate family experiments (דורש S2)
1. 3 seeds על finalists; grouped speaker development folds ו־paired evaluation; תוצאות לפי קורא וטעם.
2. H2: frozen WavLM/W2v-BERT מול P00; H4: Voice Conversion P08 מול pitch-shift-only; H6: adversarial/contrastive P09/P10 אם מספיק דוברים.
3. human review לכל synthetic accent; technical artifacts/accuracy gates; בלי tuner על final test.
4. ניתוח inter-reader variance, calibrate false accusation and coverage risk.

**Outputs:** preregistered studies, effect sizes, reader-cluster CIs, controlled ablation tables, negatives documented.

### S4 — One-shot final test and scientific reporting
1. freeze experiment selection, model SHA, thresholds + primary outcomes.
2. לפתוח את final untouched real speaker holdout **פעם אחת**.
3. לכתוב paper/repro appendix כולל כישלונות, mixed-tradition limitation, licence and ethics, uncertainty and product limitations.
4. לא להציג שיפור בסינתטי כראיה לדוברים אמיתיים.

**Outputs:** signed test ledger, paper draft, replication pack, model/dataset cards בלי raw recordings.

### S5 — Product readiness + Dvir OS reporting
1. Candidate ↔ live FastAPI regression (wav frontend/alignment/spectrogram identical).
2. עצירת טעות בטוחה: report uncertain; false accusation budget; RTL hebrew messages; latency on designated hardware; rollback.
3. לשדר summaries/hashes/metrics/hardware hours בלבד אל Dvir OS אחרי הגדרת הרשאות ומבחני פרטיות.
4. הפרדת דשבורד: academic evidence vs deployed product state vs waiting review; "reviewed" only with evidence.
5. קידום לייצור ידני בלבד.

**Outputs:** product candidate PR and release checklist, Dvir OS safe status adapter with documented access policy.

## אחריות ואיסורים
- שום תוצאה לא נחשבת "פרודקשן" כי היא ניצחה offline benchmark.
- קולות/קטינים: הסכמות, הרחקת מזהים מה־Git, מנגנון מחיקה וזכויות על samples.
- GitHub קוד/מדדי סיכום; Drive מסמכי תכנון קנוניים של Dvir OS בלבד; אודיו/מודלים כבדים בדיסק מקומי.
- אין הוצאות pay-as-you-go, אינטגרציות OAuth רחבות או התקנת launchd ללא אישור.
