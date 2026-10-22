# שער קבלת מוצר — מורה דיגיטלי מדוד

מחקר יכול להתקבל גם אם המוצר אינו מוכן. מוצר אינו נחשב אמין רק מפני ש־balanced accuracy גבוה.

## הערכת מורה למקרא

1. הפרד `taam_performed` (מה שנשמע) מ־`taam_expected` (מה שהיה צריך לקרוא) וגם מהתחזית המותנית בהקשר.
2. תן `לא בטוח — נסה שוב` כששער alignment/voice/decoding אינו מאפשר פסק דין; דווח גם coverage.
3. גרסת Taamify הפעילה אינה מוחלפת בגלל ניסוי מחקר בלי אישור מפורש.

## Gates תכנוניים לפיילוט על בני אדם

| מדד | יעד מוצע | כיצד לאמת |
|---|---|---|
| False accusation על קריאה נכונה | <=2% אחרי קביעת סף בפיתוח | recordings מסומנות ידנית דרך API חי על דוברים שונים |
| Recall על טעויות אמת | >=70% באותו budget | טעויות אותנטיות, לא label swaps |
| Abstention | להציג ולהגדיר tradeoff קביל; יעד פיילוט עד 20% | coverage-risk by class and speaker |
| Input device variance | בלי דפוסי התרעת שווא לפי מיקרופון | phones / rooms ונתוני אמת |
| End-to-end latency | למדוד p50, p95; לחתום על יעד לפני rollout | From browser to feedback, including alignment and model |
| Model compatibility | exact fingerprint of CNN/verify head/calibration/encoder | assertion before server accepts checkpoint |
| Privacy/licensing | all samples and weights permitted for this use | consent manifest and license approval |
| User feedback | messages in Hebrew RTL; clear uncertainty | manual UX and teacher evaluation |

*הספים הם יעדי תכנון, לא תוצאות שהפרויקט כבר הוכיח על קוראים חדשים.*

## תיקון ראשון שניתן לבצע עוד לפני קולות חדשים

ב־`PLAN_STAGE_A.md` מתועדים false alarms בערוצי דקדוק stress 8.1%, dagesh 5.8%, guttural 5.6%, sheva 7.5% אחרי חלונות ערוץ. יש לכייל כל ערוץ על development/calibration בלבד, ולהציג באותו גרף:
- False accusation של קורא נכון
- Sensitivity לטעות אמיתית
- Coverage/abstain
- Patient-level/reader-level split בשונות דוברים
- **Combined end-to-end false accusation** ("צריך לחזור" על תקין)

התיעוד מציין שהתרעת `צריך לחזור` הלא מוצדקת כבר ירדה ל־0 במדגם קטן לאחר mitigation, לכן אין להחליש guardrails קיימים על בסיס ממוצע ערוצים.

## מבחן E2E

יש להפעיל דרך קוד האפליקציה הנוכחי `webapp/server.py` ובפועל לשלוח WAV אל endpoint הרלוונטי, להשוות processing bytes, cropping, calibration והספים; לבדוק שאין 500 לאחר swap, וש־verify head מחובר ל־embedding space הנכון. חובה כיול לפני rollout ויכולת rollback.

## פרטיות קטינים

קולות תלמידים והקלטות אישיות אינם נשמרים ב־Git; מחייבים הרשאה מתועדת, מחיקה וגבולות שיתוף. שירותי Voice Conversion ומודלי צד שלישי כפופים לרישיון ולהסכמה. פרסום תוצאות מחקר לא כולל זיהוי דוברים או דוגמאות אודיו בלתי מורשות.
