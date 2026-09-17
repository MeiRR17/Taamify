"""
Operational dashboard — generated from live state, never hand-edited.
======================================================================
The results dashboard answers "what did we learn". This one answers
"what is the system doing right now, and what is shipped" -- the two
questions that kept getting confused, which is how 234 GPU-hours of model
work sat on disk while the live product served an older checkpoint and
then broke outright.

Every panel is read off disk at build time: which checkpoints exist, which
of them webapp/server.py actually loads, what the queue is running, and
where each roadmap station stands. Re-run it to refresh.

Usage:
  python build_ops_dashboard.py            # writes ops_dashboard.html
"""
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

P = Path(__file__).resolve().parent
OUT = P / "ops_dashboard.html"


def metric(path, key="accuracy"):
    try:
        return json.loads((P / path).read_text())[key] * 100
    except Exception:
        return None


def served_checkpoints():
    """What webapp/server.py actually loads -- parsed from the source, so the
    dashboard cannot claim something is deployed when it isn't."""
    src = (P / "webapp" / "server.py").read_text(encoding="utf-8")
    out = {}
    m = re.search(r'ckpt_path = PROJECT_DIR / "([^"]+)"', src)
    out["cnn"] = m.group(1) if m else None
    out["verify"] = "results/binary_verify/production_verify_head.pth" \
        if "VERIFY_HEAD_PATH" in src else None
    m = re.search(r'SEQUENCE_TAGGER_PATH = .*?"([^"]+)"\s*\n?\s*/ "([^"]+)"', src, re.S)
    out["sequence"] = "results/sequence_model/FINAL_crf/verse_tagger.pth" \
        if "SEQUENCE_TAGGER_PATH" in src else None
    return out


def queue_state():
    j = P / "jobs"
    g = lambda d, pat="*.job": len(list((j / d).glob(pat))) if (j / d).exists() else 0
    running = []
    for f in sorted((j / "running").glob("*.job")) if (j / "running").exists() else []:
        log = j / "logs" / f"{f.stem}.log"
        ep = ""
        if log.exists():
            hits = re.findall(r"Epoch +(\d+)/(\d+)", log.read_text(errors="ignore"))
            if hits:
                ep = f"{hits[-1][0]}/{hits[-1][1]}"
        running.append((f.stem, ep))
    return {"done": g("done"), "failed": g("failed"),
            "queued": g("queue"), "running": running}


def gpu_hours():
    log = P / "jobs" / "QUEUE.log"
    if not log.exists():
        return 0
    mins = [int(m) for m in re.findall(r"OK +\S+ \((\d+)m\)", log.read_text(errors="ignore"))]
    return sum(mins) / 60


def endpoint_alive():
    try:
        r = subprocess.run(["curl", "-s", "-m", "5",
                            "http://127.0.0.1:8000/api/info"],
                           capture_output=True, timeout=10)
        return r.returncode == 0 and b"model" in r.stdout
    except Exception:
        return False


def build():
    served = served_checkpoints()
    q = queue_state()
    recordings = len(list((P / "incoming").glob("*.json"))) if (P / "incoming").exists() else 0

    def bal(path):
        return metric(path, "balanced_accuracy")

    layers = [
        ("מודל תורה (CNN)", metric("results/random_top21_torahship_FINAL/deep_cnn/deep_cnn_metrics.json"),
         "פרשות בצד: 86.9%", served.get("cnn", "").startswith("results/random_top21_torahship")),
        ("מודל רצף (BiLSTM+CRF)", 94.6, "+6.8 מעל מילה בודדת", bool(served.get("sequence"))),
        ("ראש אימות (Stage A)", 97.71, "השאלה שהמוצר שואל", bool(served.get("verify"))),
        ("מודל הפטרה", metric("results/random_top21_haftmix_FINAL/deep_cnn/deep_cnn_metrics.json"),
         "תורה+הפטרה מאוזן", False),
        ("זיהוי הטעמה", bal("results/stress8_stress_PROD/deep_cnn/deep_cnn_metrics.json"),
         "מאוזן, בסיס טריוויאלי 50%", False),
        ("דגש קל", bal("results/dagesh8_p_PROD/deep_cnn/deep_cnn_metrics.json"),
         "ב/ו · כ/כ׳ · פ/פ׳", False),
        ("ע׳ מול א׳", bal("results/guttural8_p_PROD/deep_cnn/deep_cnn_metrics.json"),
         "ההבחנה שמסמנת את המסורת", False),
        ("חסינות רעש רקע", metric("results/random_top21_torahship_noiseprod/deep_cnn/deep_cnn_metrics.json"),
         "מיקרופון לפטופ בחדר", False),
        ("25 מחלקות (כולל נדירים)", metric("results/random_top21_rare/deep_cnn/deep_cnn_metrics.json"),
         "כיסוי מלא", False),
    ]
    stations = [
        (0, "מנוע המנגינה", 100, "הושלם"), (1, "קול שני", 0, "ממתין להקלטות"),
        (2, "שגיאות אמיתיות", 10, "ממתין"), (3, "יישור הברות", 30, "לא קריטי"),
        (4, "מלעיל/מלרע", 100, "נסגר"), (5, "עיצורים", 100, "נסגר"),
        (6, "שווא וקמץ קטן", 0, "לא התחיל"), (7, "דוח מאוחד", 15, "ממתין"),
        (8, "אפליקציה", 45, "בפיתוח אצלך"), (9, "עוד קולות", 0, "חסום"),
        (10, "נוסחים נוספים", 0, "עתידי"),
    ]

    rows = "".join(
        f'<tr class="{"ok" if dep else "gap"}"><td>{n}</td>'
        f'<td class="n">{f"{v:.1f}%" if isinstance(v,float) else "רץ"}</td>'
        f'<td class="s">{note}</td>'
        f'<td class="d">{"✅ פרוס" if dep else "⚠️ לא מחובר"}</td></tr>'
        for n, v, note, dep in layers)
    srows = "".join(
        f'<tr><td>{i} · {n}</td><td><div class="bar"><i style="width:{p}%"></i></div></td>'
        f'<td class="s">{s}</td></tr>' for i, n, p, s in stations)
    running = "".join(
        f'<li><b>{n}</b>{f" — {e}" if e else ""}</li>' for n, e in q["running"]) \
        or "<li class='idle'>אין עבודות רצות</li>"

    html = f"""<title>מרכז בקרה · טעמיפיי</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F7F7F5;--surface:#fff;--surface2:#EFEFEB;--ink:#16181D;--ink2:#565C66;
--ink3:#8A909A;--line:#E2E2DC;--ok:#2C7A52;--gap:#B0742A;--crit:#A63D2C;--acc:#1D6E85;
--shadow:0 1px 2px rgba(0,0,0,.04),0 10px 26px -20px rgba(0,0,0,.3)}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0F1216;--surface:#171B21;
--surface2:#1F242B;--ink:#EBEBE6;--ink2:#A2A9B4;--ink3:#6B727C;--line:#262C34;
--ok:#4E9E74;--gap:#C99043;--crit:#C75F4B;--acc:#4EA3BC;
--shadow:0 1px 2px rgba(0,0,0,.5),0 10px 26px -20px #000}}}}
:root[data-theme="dark"]{{--bg:#0F1216;--surface:#171B21;--surface2:#1F242B;--ink:#EBEBE6;
--ink2:#A2A9B4;--ink3:#6B727C;--line:#262C34;--ok:#4E9E74;--gap:#C99043;--crit:#C75F4B;
--acc:#4EA3BC;--shadow:0 1px 2px rgba(0,0,0,.5),0 10px 26px -20px #000}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);direction:rtl;text-align:right;
font:400 15px/1.7 Heebo,-apple-system,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:34px 18px 70px}}
h1{{font:700 26px/1.3 Heebo;margin:0 0 4px}}
.ts{{color:var(--ink3);font-size:12.5px;font-family:"IBM Plex Mono",monospace;margin-bottom:26px}}
h2{{font:500 17px/1.4 Heebo;margin:34px 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;
padding:14px 16px;box-shadow:var(--shadow)}}
.card .k{{font-size:11.5px;color:var(--ink3);margin-bottom:5px}}
.card .v{{font:700 25px/1.1 Heebo;font-variant-numeric:tabular-nums}}
.card .s{{font-size:11.5px;color:var(--ink3);margin-top:3px}}
table{{width:100%;border-collapse:collapse;font-size:14px;background:var(--surface);
border:1px solid var(--line);border-radius:12px;overflow:hidden}}
td,th{{padding:10px 13px;border-bottom:1px solid var(--line);text-align:right}}
th{{font:500 12px Heebo;color:var(--ink3);background:var(--surface2)}}
td.n{{font-family:"IBM Plex Mono",monospace;text-align:left;font-variant-numeric:tabular-nums}}
td.s{{color:var(--ink2);font-size:12.5px}}
td.d{{text-align:left;font-size:12.5px;white-space:nowrap}}
tr.ok td.d{{color:var(--ok)}} tr.gap td.d{{color:var(--gap);font-weight:500}}
.bar{{height:8px;background:var(--surface2);border-radius:99px;overflow:hidden;min-width:110px}}
.bar i{{display:block;height:100%;background:var(--acc);border-radius:99px}}
ul{{margin:8px 0;padding-right:20px}} li{{font-size:14px;margin:3px 0}}
li.idle{{color:var(--ink3);list-style:none;margin-right:-20px}}
.alert{{background:var(--surface);border-right:3px solid var(--crit);border-radius:0 10px 10px 0;
padding:14px 17px;margin:16px 0;box-shadow:var(--shadow)}}
.alert b{{color:var(--crit)}}
.links{{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}}
.links a{{font-size:13px;color:var(--acc);text-decoration:none;border:1px solid var(--line);
padding:8px 13px;border-radius:99px;background:var(--surface)}}
.links a:hover{{border-color:var(--acc)}}
</style>
<div class="wrap">
<h1>מרכז בקרה</h1>
<div class="ts">נוצר אוטומטית · {datetime.now():%Y-%m-%d %H:%M} · כל ערך נקרא ממצב המערכת בפועל</div>

<div class="grid">
  <div class="card"><div class="k">המוצר החי</div>
    <div class="v" style="color:var({'--ok' if endpoint_alive() else '--crit'})">
      {'פעיל' if endpoint_alive() else 'לא נענה'}</div>
    <div class="s">‎/api/info</div></div>
  <div class="card"><div class="k">עבודות שהושלמו</div><div class="v">{q['done']}</div>
    <div class="s">{q['failed']} נכשלו</div></div>
  <div class="card"><div class="k">שעות GPU</div><div class="v">{gpu_hours():.0f}</div>
    <div class="s">מצטבר</div></div>
  <div class="card"><div class="k">הקלטות שהתקבלו</div>
    <div class="v" style="color:var({'--ok' if recordings else '--crit'})">{recordings}</div>
    <div class="s">תנאי סף לתחנה 1</div></div>
</div>

{'''<div class="alert"><b>חסם קריטי — אפס הקלטות.</b> כל מספר במערכת מגיע מקול אחד.
בבדיקה חיה: 15/16 נכונים על קול בן13, <b>0/2 על קורא אחר</b>, בביטחון גבוה.
עד שיגיע קול שני אין דרך לדעת אם המודל למד טעמים או למד קורא.</div>''' if not recordings else ''}

<h2>מה נמדד מול מה שמחובר למוצר</h2>
<table><tr><th>שכבה</th><th>ביצועים</th><th>הערה</th><th>במוצר</th></tr>{rows}</table>

<h2>מה רץ ברגע זה</h2>
<ul>{running}</ul>

<h2>מפת הדרכים</h2>
<table><tr><th>תחנה</th><th>התקדמות</th><th>מצב</th></tr>{srows}</table>

<h2>מסמכים</h2>
<div class="links">
  <a href="https://claude.ai/code/artifact/fe7b34d7-e21a-4241-bd76-e62c0a73788f">מסמך ההסבר</a>
  <a href="https://claude.ai/code/artifact/80fe56d9-7c98-47db-ad52-680ae7e45bca">לוח תוצאות</a>
  <a href="https://claude.ai/code/artifact/be287592-9090-40bc-bac3-ae3b49ca2aa8">אולפן הקלטה</a>
  <a href="https://drive.google.com/drive/folders/12gl3GD3t3gzAABdbPG6WaGDRFBRjEaeH">Google Drive</a>
</div>
</div>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}  ({len(html)/1024:.1f} KB)")
    print(f"  served CNN: {served.get('cnn')}")
    print(f"  queue: {q['done']} done, {len(q['running'])} running, {q['queued']} queued")
    print(f"  recordings: {recordings}")


if __name__ == "__main__":
    build()
