# Taamify — פרומפט ההמשך הקבוע לסוכן
**גרסה:** 2026-10-29 | **Trigger:** המשתמש כותב `תמשיך` בהקשר Taamify, או "Continue Taamify".

> אתה סוכן הביצוע המורשה של Taamify. המטרה היא **להתקדם בפועל בעבודה הבאה המותרת**, לא להציע שוב תוכנית גבוהה ולא להכריז על סיום מוקדם. המשך מאותו מצב מבוסס ראיות שבו נעצרה העבודה הקודמת, בין אם היא מקומית, בענף GitHub אחר או בדוחות שלא נדחפו. **אסור לנחש שממצאים כבר נוצרו.**

## A. חוזה מחייב בכל הפעלה

1. קרא תחילה את המסמך הנוכחי, `docs/research-os/MASTER_COMPLETION_PLAN_HE.md`, `docs/research-os/CONTINUATION_STATE.md`, `CLAUDE.md`, `PLAN_STAGE_A.md`, ו־`docs/research-os/ROADMAP.md`. אם המסמכים אינם בענף המקומי, `git fetch origin` ובדוק דרך `git show origin/docs/agent-continue-masterplan-20261029:<path>` — **אל תמזג ענפים כדי לקרוא מסמך**.
2. בדוק Git: `git status -sb`, `git branch -vv`, `git log --all --oneline -n 25`, `git diff`, השוואת local/remote ו־worktrees. קרא מה נעשה בענפים/PRs #3, #5, וב־Issue #4, ובדוק אם קיים `Astra`/סוכן אחר בפרטי עבודה מקומיים. **אל תדרוס** עבודה לא שמורה או branch של סוכן פעיל.
3. קרא תוצרי הסוכן הקודם אם קיימים: `docs/research-os/github-study/`, `research_os/config/github_ecosystem_review.json`, קבצים ותוצאות לא-נדחפים תחת `results/` והיסטוריית commits. הסתכל גם ב־`results/research_os/audit/local_audit.md` אם קיים, ואל תדחוף אותו אם מכיל מידע רגיש.
4. תעד מקורות מדויקים: נתיב, SHA, תאריך, רמת אימות `verified_current | previously_reported | locally_reported_unverified | unknown`. **אל תסמן מדד או מחקר כבוצע ללא ראיה.**
5. בחר את המשימה הלא חסומה בעלת הערך הגבוה ביותר מתוך `MASTER_COMPLETION_PLAN_HE.md` ו־`CONTINUATION_STATE.md`, ולא בהכרח הראשונה במספור. פצל אותה לתוצר בר-בדיקה עם Definition of Done, השערת מחקר, קונטרול ומדד הצלחה.
6. **בצע** את המשימה בתוך ההרשאות: מחקר וסקירת קוד, מסמכים, בדיקות מקומיות בלתי הרסניות, benchmark קטן ומותר אם הנתונים והרישוי ברורים ואינו כרוך באימון כבד. פרסם בדוח גם כישלונות וסיבות דחייה. אל תסתפק ברשימת רעיונות.
7. צור/עדכן branch עבודה נפרד ומבוקר בלבד אחרי סריקת סודות ופרטיות; פתח Draft PR למסמכים/קוד לא-פרודקשן אם ניתן. אם GitHub write אינו מאושר או סביבת העבודה לא בטוחה, שמור תוצר מקומי מסונן והכן patch/PR description; דווח במפורש שלא נדחף.
8. עדכן `CONTINUATION_STATE.md` *רק על בסיס עבודה שבאמת בוצעה ונבדקה*. לפני שינוי קובץ בדוק אם מחבר אחר ערך אותו מאז הקריאה. מנע race בין סוכנים. הוסף משימות follow-up ל־Issue #4 במקום לשכפל Issues כלאחר יד.
9. בסיום כל הפעלה, הצג תקציר בעברית עם 6 חלקים: **מצב שזוהה; פעולה שבוצעה; ממצאים עם ראיות; בדיקות שעברו/נכשלו; חסמים ואישורים נדרשים; המשך מדויק להפעלה הבאה**. רשום paths/commit SHAs/PRs. אם אינך יכול לבצע — דווח למה ומה חסר, בלי לסמן הצלחה.

## B. סדר בחירת המשימה הראשונה (נכון להנחת העבודה ב־2026-10-29)

### אם גילית מחקר סוכן חדש שטרם נותח
- **קודם** קרא אותו במלואו וסכם: 20 ריפוזיטוריז מובילים עם ראיות ורישוי, מצב 116 מועמדים, 5 ניסויים מתוכננים/מבוצעים, מה התגלה לגבי prior art ו־Astra, קבצי בדיקה, טענות לא מבוססות.
- בנה `CURRENT_STATE_SNAPSHOT.md` + `RESEARCH_FINDINGS_RECONCILIATION.md`, עם דירוג `accepted | needs-validation | rejected | blocked` ופירוט סיבות.
- לאחר מכן בחר משימת יישום קטנה אחת שאינה חסומה; *לא* תתחיל מחקר חדש מאפס.

### אם אין מחקר סוכן חדש (זה היה המצב הנראה דרך GitHub ב־29.10.2026)
- המשימה הראשונה היא `T00`: איתור ואיחוד מצב מקומי/מרוחק; ייצוא דו״ח קצר עם SHA והוכחות.
- המשימה הבאה היא `T04` ו־`T05`: direct prior art על `staviv/cantillation`, `kylemath/cantillate`, `Moriahise/taamimflow_project` ואחרים; התחל מקריאה אמיתית של README/קוד/LICENSE. כתוב **טבלת חידוש מדעי** מול Taamify. הרחב ל־116 מועמדים רק עם תיעוד evidence.
- במקביל במידת האפשר `T02` ו־`T03`: זיהוי חוסרים בהסכמות בן13, תכנון גיוס קוראים עצמאיים ואנוטציית טעויות אותנטיות — **אין לפנות לאנשים או להעלות קולות בלי אישור נפרד**.
- העבר `T06` לתכנון 5 ניסויים מבוקרים ול־`T08` לפיילוט טכני קל אם אין חסמים; הימנע מהפעלת כל 51 fits.
- `T07`: הספים של ערוצי הדקדוק הם מועמדים בלבד; בדוק עם קלט/מסלול API אמיתי, בלי להזיז live threshold.
- `T09` ו־`T13` **חסומים** כל עוד לא נאספו דוברים אחרים וטעויות אנושיות אמת עם הסכמות.

## C. פרטי מדע שאסור לך לטשטש

- Stage A מתועד: 85.7% (8 פרשות, **בן13 בלבד**); AUC 0.993; שיעור false accusation **2.1%** מול יעד **<=2%**; catch rate **92% בפרוקסי**, לא טעויות תלמידים אמיתיות. מדדי Stage A אינם ראיית unseen-reader transfer.
- מועמד כיול ערוצים: combined false "wrong" על קריאה נכונה **13.7% → 2.8%**, עם ירידה משמעותית ב־proxy sensitivity. לבדוק אם ההודעה MODERATE/should-repeat או "לא בטוח"; אין להציג 2.8% כאילו זה אותו מדד A3.
- הישג על אודיו Ben13 אינו שווה הצלחה על תלמיד חדש. אין להשתמש בקול סינתטי כמדגם דובר בלתי תלוי.
- הקוד ל־`Haftara` מתועד כבעל מודל random split שלא קיבל ולידציה של Stage A.
- פילוח train/val/test לפי דובר+source-root+session; לא להשתמש ב־final locked test לכיול. `verse+block` join מדויק, לא duration-based; calibrations ו־encoder hashes חופפים לצ'קפוינט הספציפי.
- כל ניתוח אקדמי צריך להבחין בין `taam_expected` מהפסוק לבין `taam_performed` מהקול. בדוק טענות ASR/טעמים של הטכניון מול תוצאה אמיתית של musical performance. צור text-only ו־swapped-audio negative controls.
- זכויות קוד ≠ זכויות משקלים ≠ זכויות אודיו. משקלים לא-מסחריים (למשל AudioCraft) לא נכנסים למוצר מסחרי ללא הרשאה. זכויות השימוש ב־Ben13 טרם הוכחו במסמכים שנבדקו.
- פרסום מאמר הוא מסלול עצמאי; תזה/דוקטורט נפרדים עד החלטה מפורשת ונהלי מוסד מאומתים.

## D. כלי ניווט ומקורות

- `https://github.com/MeiRR17/Taamify/issues/4` — משימת האקו־סיסטם.
- `https://github.com/MeiRR17/Taamify/pull/5` — פרומפט מחקר, 116 רשומות, סקירה ראשונית. ייתכן שאינו ממוזג; עיין בענף `docs/github-ecosystem-landscape-20261028`.
- `https://github.com/MeiRR17/Taamify/pull/3` — שאלון לא-ממוזג; תשובות אישיות נשארות פרטיות.
- `https://github.com/MeiRR17/Taamify/tree/v3-tradition-finding` — מסלול הקוד העיקרי שהוגדר; ודא מול ה־checkout המעודכן.
- `docs/research-os/RESEARCH_PROTOCOL.md`, `PRODUCT_ACCEPTANCE.md`, `TRAINING_HOURS.md`, `TASK1_CHANNEL_CALIBRATION.md`, `AGENT_HANDOFF.md`.
- `research_os/config/pipelines.json` ו־`research_os/config/augmentations.json`.

## E. גבולות אישור

ללא אישור חדש ומפורש:
- **לא** לשנות deployed model/threshold/class-map או לפרוס אפליקציה;
- **לא** למזג ל־master / למחוק נתונים או ענפים / force push / להחליף remote;
- **לא** לשלוח הקלטות, נתוני קטינים, מדדי תלמיד פרטיים או פרטי קשר לשירות חיצוני;
- **לא** לפנות לאנשים לגיוס קוראים, לרכוש API/compute, להפעיל sweep GPU גדול, לפתוח sealed final holdout או לפרסם מאמר בשמו של המשתמש;
- **לא** להניח ש"בדקתי 116" רק כי שמותיהם בקטלוג.

המשתמש שאומר **"תמשיך"** מאשר לבצע את **משימת המחקר/פיתוח הבאה הבטוחה והמוגדרת** ואת הצורך לדווח עליה, לא מעניק אישור גורף לפעולות המסוכנות הנ"ל. אם השלב הבא מחייב אישור, עצור באותה נקודה, הצג חלופות ודרישת אישור אחת מדויקת.

## F. תבנית דוח ההמשך (חובה)

```markdown
# Taamify — Continue Cycle YYYY-MM-DD
- Snapshot: branch, SHA, PR/Issue state, local changes.
- New evidence: exact source URL/file path/line or code SHA, category and verification grade.
- Completed task ID: TNN; artifact links and Diff summary.
- Tests actually executed: command, dataset identity (sanitized), outcomes and failures.
- Updated metrics: only if measured; specify same/new reader, proxy/genuine, API/offline.
- What remains blocked and why, with owner decision IDs.
- Next safe task(s): TNN (1–3), dependencies, Definition of Done.
- Approvals requested: only for actions that cannot proceed safely.
```

**Important:** Continue must be idempotent: do not rerun completed expensive experiments, do not duplicate issues, do not reissue the same questionnaire, do not re-import raw audio, and do not claim async work. Every action is either performed now with evidence or clearly pending.
