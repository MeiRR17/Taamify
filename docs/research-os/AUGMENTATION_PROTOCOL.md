# Augmentation study — אוגמנטציה מבוקרת, לא עוד דאטה לא מבוקר

**שאלה:** האם אוגמנטציה משפרת הכללה של טעמים ב־unseen genuine reader, ולא רק משפרת דיוק על גרסאות נגזרות של אותה הקלטה?

המפתח הוא compare under matched training budget. **Pitch shift כבר נבדק ב־Taamify והראה עמידות לגובה קול.** כל שיטה אחרת מחויבת להשתוות לבקרת Pitch Shift באותם הקלטות/מודל/זרעים/חלוקת דוברים.

## קטלוג אוגמנטציות והבקרות (מפרט: `research_os/config/augmentations.json`)

| ID | טכניקה | מה היא משנה | בקרת תקינות | עדיפות |
|---|---|---|---|---|
| A00 | Pitch shift | F0 מוחלט | שמירת מרווחים וקונטור הטעם | קיימת ובסיס השוואה |
| A01 | Room impulse responses | חדר, הדהוד | ללא שינוי במלעיל/מלרע/גבולות | גבוהה |
| A02 | Noise profiles | רעשי רקע + SNR | בלי masking שמסתיר טעם | גבוהה |
| A03 | Time stretch | קצב/משך | בלי שינוי pitch ובלי איבוד הברות | בינונית |
| A04 | Formant/VTL perturbation | מבנה קול, לא תוכן | F0 וטקסט נשמרים | בינונית |
| A05 | Codec/mobile channel | מיקרופון/דחיסה | אין clipping ואין label corruption | גבוהה למוצר |
| A06 | F0-controlled voice conversion | גוון קול יעד | הסכמה + קונטור+הברות+מבנה טעם | מחקר מותנה |
| A07 | SpecAugment | מיסוך תדר/זמן | לא למחוק אזורי accent משמעותיים | קיים |
| A08 | Mixup | ערבוב אודיו | עלול ליצור מנגינה ללא label תקף | לא להפעיל כברירת מחדל |

## מחקר שימור F0, הטעמה ופונמות

לפני שמוסיפים `generated_audio` לרכבת האימון:
1. אל תסיק ש"נשמע כמו דובר אחר" פירושו שהטעם נשמר.
2. מדוד על אותו מקור F0 יחסית בחצאי טונים אחרי octave-correction, כיוון contour, מספר turning points, duration/relative vowel timings, speech intelligibility.
3. דגום ידנית זוגות מקור ↔ נגזרת, עיוור למערכת שהפיקה, ותייג `unchanged_taam | changed_taam | ambiguous`. מגבילים אימון ל־`unchanged_taam`.
4. זיהוי מטעה של מקור סינתטי: train a classifier real vs synthetic on embeddings; אם קל להבדיל, חשוד שהמודל למד artifacts במקום טעמים.
5. השווה תועלת `A00` בלבד מול `A00+A0i` על **אותם** דוברים חדשים אמיתיים.

## זליגת מקור ודובר

סינתטי שומר `root_recording_id` של המקור, ואף פעם לא עובר ל־validation/test אחרת מהמקור. נוסף על כך, `voice_reference_speaker_id` חייב להשתייך **רק** ל־train. לעולם לא להשתמש בדובר מבחן כ"קול יעד" להמרה עבור דוגמאות אימון. אסור להשתמש בדובר בעל קול ללא רישיון/הסכמה מתאימה.

## ניסויים חשבוניים

- Feasibility: 15–30 דקות audio חתוכות מאומתות, להשוות pitch-tracking real↔synthetic, איכות alignment, failure rate, צריכת מאיץ.
- Pilot: P00 + A00 מול P00 + A00 + A01/02/03/04/05, 3 seeds אחרי feasibility אם קיים effect.
- Voice conversion confirmatory: P00 + A00 מול P00 + A00 + A06, ללא augmentation imbalance; לבצע אך ורק לאחר קוראים חדשים, ובדיקת F0 והלכות רישוי.
- Metadata: source/root IDs, generator/revision, transforms + seed, consent scope, licence, F0 audit, annotator disagreement; no raw audio in Git.

**הפרדה מדעית:** אוגמנטציה אינה דובר עצמאי; 100 שכפולי קול סינתטיים אינם 100 נבדקים.
