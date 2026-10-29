# קליטת הקלטות עם תוויות ביצוע אנושיות

הקלטת משתמש אינה הופכת לדוגמת אימון מפני שהטקסט המצופה ידוע או מפני שהמודל אישר אותה. מסלול הקליטה משתמש כעת בהסכמה מפורשת, מזהי קורא ומפגש בדויים, נוסח ספרדי־ירושלמי ותווית ביצוע שנבדקה בידי אדם.

## סדר עבודה

1. מקליטים במסך התרגול עם אישור השמירה המקומית והאפשרות לשימוש עתידי באימון. ההקלטה והיומן נשארים מקומיים בתיקייה `app_assets/collected_phrases/` המוחרגת מ־Git.
2. מפיקים תבנית תיוג חדשה:

```bash
venv/bin/python Data_Prep/voice_intake.py \
  --review-template results/research_os/intake/pending_review.json
```

3. מתייג אנושי מאזין וממלא `status`, `reviewed_by`, `reviewed_at` עם אזור זמן, `label_source`, `reading_quality`, `performed_words` ו־`performed_taamim`. התבנית ממלאת את טביעות המקור בלבד; היא אינה מעתיקה תחזיות מודל לתווית אדם.
4. שומרים את ההערות בקובץ תיוג נפרד ומפיקים אישור חדש:

```bash
venv/bin/python Data_Prep/voice_intake.py \
  --decisions results/research_os/intake/human_review.json \
  --promote --out results/research_os/intake/approval_v1.json
```

5. בודקים את כל מקורות האישור לפני חילוץ מאפיינים:

```bash
venv/bin/python Data_Prep/ingest_collected_phrases.py \
  --decisions results/research_os/intake/human_review.json \
  --approval results/research_os/intake/approval_v1.json --validate-only
```

6. יוצרים תמונת מאפיינים בתיקייה חדשה בלבד:

```bash
venv/bin/python Data_Prep/ingest_collected_phrases.py \
  --decisions results/research_os/intake/human_review.json \
  --approval results/research_os/intake/approval_v1.json \
  --out-dir features_v3_reviewed/intake_v1
```

## חוזה התיוג

דוגמה מבנית, **לא תווית אמיתית או אישור להקלטה קיימת**:

```json
{
  "schema_version": 2,
  "clips": {
    "clip-id-from-template": {
      "status": "approved",
      "label_source": "human-reviewed",
      "reviewed_by": "reviewer_001",
      "reviewed_at": "2026-10-28T12:00:00+02:00",
      "reading_quality": "clear",
      "audio_sha256": "copy-the-unchanged-hash-from-the-template",
      "record_sha256": "copy-the-unchanged-hash-from-the-template",
      "performed_words": ["המילה שנבדקה"],
      "performed_taamim": ["Tipecha"]
    }
  }
}
```

`label_source` יכול להיות `human-reviewed` או `adjudicated`. שם מתייג הוא מזהה בדוי, לא שם אישי. לכל מילה ניתן טעם מתוך 21 המחלקות הנתמכות; `null` משאיר מילה מחוץ לאימון כשהטעם לא הוכרע. לפחות טעם אחד בקטע חייב להיות מוכרע.

אם המילים עצמן הוחלפו/הושמטו, הרשומה נשארת מחוץ למסווג התקין ומקבלת סיבת דחייה `performed_words_differ_requires_error_dataset`. היא חשובה למאגר טעויות, אך זקוקה ליישור ובדיקה ייעודיים. אם המילים זהות והטעם שבוצע שונה מהמצופה, התווית לאימון היא הטעם **שנשמע ואושר אנושית**, ושדה `taam_expected` נשמר בנפרד.

## מה נחסם

- קובץ אישור חסר, ישן או שונה, אודיו ששונה, שינוי ברשומת המקור או בתיוג אחרי האישור.
- נוסח לא מזוהה או לא מתאים, מזהה קורא/מפגש חסר, הסכמה חסרה/לא מתאימה או היעדר הרשאה לשימוש באימון.
- אישור המבוסס רק על `statuses=ok`, תג `gold` או מחרוזת `verified` ישנה ללא תיעוד ביצוע.
- דריסת תיקיית מאפיינים קיימת או ערבוב אוטומטי עם מאגר בן13.

כל סיבה נספרת בדוח. הקבצים הגולמיים נשמרים ללא שינוי, ולא הופעל במסלול הזה אימון או שינוי של מודל המוצר.
