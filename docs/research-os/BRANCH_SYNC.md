# שילוב תוכנית המחקר עם 53 הקומיטים המקומיים — Runbook בטוח

**מצב מתועד:** `v3-tradition-finding` המקומי מקדים את `origin/v3-tradition-finding` ב־53 קומיטים. הענף `docs/taamify-research-os-handoff-20261022` נוצר **מהענף המרוחק**, ולכן אינו מכיל כרגע את 53 הקומיטים.

## סדר ביצוע מומלץ עבור הסוכן המקומי

```bash
git status --short
git branch --show-current
git fetch origin --prune
git log -1 --format='%h %ad %s' --date=iso
git rev-list --left-right --count origin/v3-tradition-finding...v3-tradition-finding
git diff --stat origin/v3-tradition-finding...v3-tradition-finding
```

אם יש קבצים מקומיים בלתי שמורים, עצור ואל תדרוס אותם. צור **גיבוי מקומי** לפני כל merge/cherry-pick:

```bash
git switch v3-tradition-finding
git branch backup/v3-before-research-os-20261022
```

לפני Push של ההיסטוריה המקומית, בדוק לאורך הקומיטים ולא רק ב־HEAD שאין הקלטות תלמידים/קולות מורשים, `.env`, מפתחות, `incoming/`, `app_assets/collected_attempts/`, `features_*/`, checkpoints כבדים או נתונים מזהים. סריקת `git log --name-only origin/v3-tradition-finding..HEAD` ומבקרת סודות נפרדת היא תנאי.

**דחיפה למרוחק:** לאחר ביקורת היסטוריה ואישור מפורש של הבעלים ובהנחה שהענף עדיין מתקדם באופן fast-forward, לבצע `git push origin v3-tradition-finding` (ללא force). אין לבצע push על פי המלצה של סוכן ללא הרשאה.

**לצרף את התוכנית למקומי**, רק אחרי קבלת המצב המקומי ושמירת העבודה:
```bash
git fetch origin
git switch v3-tradition-finding
git merge --no-ff origin/docs/taamify-research-os-handoff-20261022
```
אם הענף המרוחק כולל רק את ענף המחקר המשותף הישן בתוספת מסמכים, והיסטוריות תואמות, ה־merge ייצור מיזוג נקי בדרך כלל. אם יש קונפליקט, יש **לעצור** ולבדוק במקום להשתמש ב־`-X ours` או `-X theirs` בצורה גורפת. חלופה למיזוג: אחרי אישור אנושי, cherry-pick של הקומיטים המסוימים המוסיפים את המסמכים.

## נקודות רגישות

- מקומית ייתכנו כבר `CLAUDE.md`, `PLAN_STAGE_A.md`, `run_queue.sh` ו־`Data_Prep/holdout.py`; הם מקור האמת לעומת ה־README הישן מרחוק. **אין להחליף** אותם במסמכי התכנון הישנים.
- אל תבצע `git reset --hard`, `git clean -fd`, `git push --force` או שינוי branch default כדי "לפתור" פערים.
- אל תמזג ל־`master` כחלק מפעולת bootstrap. בקש סקירה של diff + tests.
- אם מעדיפים לא לדחוף את 53 הקומיטים עדיין, ניתן למזג מקומית את ענף המסמכים ולעבוד שם; תצוגת GitHub במקביל עדיין תהיה חסרה את העבודה המקומית.

## הצלחה

הסוכן מדווח על `git log -5`, נתיב מסמך זה ב־checkout, קיום `PLAN_STAGE_A.md`, בדיקת `git status`, והאם קובצי המקור וה־checkpoints עומדים בבקרות פרטיות; אין התחלת אימון עד שכל שערי S0 עוברים.
