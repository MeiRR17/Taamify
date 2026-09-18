"""
Render tikkun_data.json as the reference page.
==============================================
The markdown version is for reading in the repo; this is the one a teacher or
a rabbi is handed. It shows what markdown cannot: the contour as an actual
curve with its spread drawn around it, so that "measured" is visible rather
than asserted.

Regenerate after build_tikkun.py -- the page is a view, never a source.

Usage:
  python Data_Prep/build_tikkun_page.py
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent
DATA = P / "results" / "melody_research" / "tikkun_data.json"
OUT = P / "results" / "melody_research" / "tikkun_soferim.html"

HE = {"strict": "משנה משמעות", "moderate": "עלול לשנות משמעות",
      "lenient": "אינו משנה משמעות"}


def curve_svg(curve, sd, w=300, h=88, pad=6):
    """One series, its own scale, with the across-occurrence spread as a band.

    The band is the honest part: for most ta'amim the spread is wider than
    the mean shape, which says the contour is a tendency and not a template.
    """
    if not curve:
        return ""
    n = len(curve)
    lo = min(c - s for c, s in zip(curve, sd))
    hi = max(c + s for c, s in zip(curve, sd))
    span = max(hi - lo, 0.5)

    def pt(i, v):
        x = pad + (w - 2 * pad) * i / (n - 1)
        y = h - pad - (h - 2 * pad) * (v - lo) / span
        return f"{x:.1f},{y:.1f}"

    top = " ".join(pt(i, c + s) for i, (c, s) in enumerate(zip(curve, sd)))
    bot = " ".join(pt(i, c - s) for i, (c, s) in
                   reversed(list(enumerate(zip(curve, sd)))))
    line = " ".join(pt(i, c) for i, c in enumerate(curve))
    mid = h - pad - (h - 2 * pad) * (0 - lo) / span
    return (
        f'<svg class="curve" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="עקומת המנגינה">'
        f'<line class="zero" x1="{pad}" y1="{mid:.1f}" x2="{w-pad}" '
        f'y2="{mid:.1f}"/>'
        f'<polygon class="band" points="{top} {bot}"/>'
        f'<polyline class="mean" points="{line}"/>'
        f'</svg>')


def bar(label, pct):
    return (f'<div class="nb"><span class="nb-l">{label}</span>'
            f'<span class="nb-bar"><i style="inline-size:{pct:.0f}%"></i>'
            f'</span><span class="nb-p">{pct:.0f}%</span></div>')


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    he_of = {k: v["he"] for k, v in d.items()}
    order = sorted(d, key=lambda t: (d[t]["rank"] or 9, -d[t]["count"]))
    total = sum(v["count"] for v in d.values())
    checkable = [t for t in order if d[t]["checkable"]]

    nav = "".join(
        f'<a href="#t-{t}"><span class="nav-sign">א{d[t]["sign"]}</span>'
        f'<span class="nav-name">{d[t]["he"]}</span>'
        f'<span class="nav-n">{d[t]["count"]:,}</span></a>'
        for t in order)

    cards = []
    for t in order:
        v = d[t]
        role = "מפסיק" if v["rank"] else "משרת"
        sev_cls = v["severity"]
        nbs = ""
        if v["follows"]:
            tot = sum(n for _, n in v["follows"])
            a, n = v["follows"][0]
            nbs += bar(f"אחריו · {he_of.get(a, a)}", n / tot * 100)
        if v["preceded_by"]:
            tot = sum(n for _, n in v["preceded_by"])
            a, n = v["preceded_by"][0]
            nbs += bar(f"לפניו · {he_of.get(a, a)}", n / tot * 100)

        if v["checkable"]:
            mel = (curve_svg(v["curve"], v["curve_sd"]) +
                   f'<p class="contour">{v["contour_he"]}</p>'
                   f'<p class="meta-line">נמדד על {v["curve_n"]} מופעים · '
                   f'עקביות {v["consistency"]:.2f}</p>')
        else:
            mel = ('<p class="unchecked">מעט מדי הקלטות כדי לקבוע צורה. '
                   'הטעם הזה נלמד ונשמע — ואינו מקבל ציון.</p>')

        pos = v["position_mean"]
        pos_line = ""
        if pos is not None:
            where = ("בתחילת הפסוק" if pos < 0.3 else
                     "בסוף הפסוק" if pos > 0.7 else "באמצע הפסוק")
            pos_line = (f'<div class="pos"><span class="pos-track">'
                        f'<i style="inset-inline-start:{(1-pos)*100:.0f}%">'
                        f'</i></span>'
                        f'<span class="pos-t">{where} · '
                        f'{pos*100:.0f}% מאורכו</span></div>')

        cards.append(f"""
<article class="card" id="t-{t}">
  <header class="card-head">
    <div class="glyph" aria-hidden="true">א{v['sign']}</div>
    <div class="titles">
      <h2>{v['he']}</h2>
      {f'<p class="seph">בפי ספרדים: {v["sephardi"]}</p>' if v.get('sephardi') and v['sephardi'] != v['he'] else ''}
      <p class="facts">
        <span class="chip role-{'dis' if v['rank'] else 'con'}">{role}{f" · דרגה {v['rank']}" if v['rank'] else ''}</span>
        <span class="chip sev sev-{sev_cls}">{HE[sev_cls]}</span>
        <span class="count">{v['count']:,} מופעים · {v['share']*100:.2f}%</span>
      </p>
    </div>
  </header>
  <p class="role-he">{v['role_he']}</p>
  {pos_line}
  <div class="split">
    <section class="mel">
      <h3>הסלסול</h3>
      {mel}
    </section>
    <section class="syn">
      <h3>השכנים</h3>
      {nbs or '<p class="meta-line">—</p>'}
    </section>
  </div>
</article>""")

    html = f"""<title>תיקון סופרים</title>
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@500;700&family=Heebo:wght@400;500;700&family=Noto+Serif+Hebrew:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#FBF9F4; --ink:#16181C; --ink2:#545A64; --ink3:#8A8F98;
  --rule:#E3DED2; --card:#FFFDF8;
  --melody:#1B6E88; --melody-soft:#1B6E8826;
  --gold:#A87C2A; --clay:#B3543A; --green:#2E7D57;
  --sans:"Heebo",system-ui,sans-serif;
  --serif:"Frank Ruhl Libre",Georgia,serif;
  --bible:"Noto Serif Hebrew","Frank Ruhl Libre",serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#12141A; --ink:#ECEAE4; --ink2:#A2A9B4; --ink3:#767D88;
    --rule:#262A33; --card:#191C23;
    --melody:#4BA3BE; --melody-soft:#4BA3BE2E;
    --gold:#C99A42; --clay:#C8674E; --green:#4C9E76;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#12141A; --ink:#ECEAE4; --ink2:#A2A9B4; --ink3:#767D88;
  --rule:#262A33; --card:#191C23;
  --melody:#4BA3BE; --melody-soft:#4BA3BE2E;
  --gold:#C99A42; --clay:#C8674E; --green:#4C9E76;
}}
* {{ box-sizing:border-box }}
body {{
  background:var(--paper); color:var(--ink);
  font-family:var(--sans); direction:rtl;
  font-size:15px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ display:grid; grid-template-columns:230px minmax(0,1fr);
  gap:44px; max-inline-size:1180px; margin-inline:auto;
  padding:0 28px 96px; align-items:start; }}
@media (max-width:900px) {{ .wrap {{ grid-template-columns:1fr; gap:0 }} }}

header.top {{ max-inline-size:1180px; margin-inline:auto;
  padding:72px 28px 40px; }}
header.top h1 {{ font-family:var(--serif); font-weight:700;
  font-size:clamp(2.5rem,6vw,4rem); line-height:1.08; margin:0 0 .35em;
  letter-spacing:-.015em; text-wrap:balance; }}
header.top .sub {{ font-size:1.12rem; color:var(--ink2);
  max-inline-size:62ch; margin:0 0 1.6em; }}
.stats {{ display:flex; flex-wrap:wrap; gap:0; border-block:1px solid var(--rule);
  padding-block:18px; }}
.stat {{ padding-inline:26px; border-inline-start:1px solid var(--rule); }}
.stat:first-child {{ border:0; padding-inline-start:0 }}
.stat b {{ display:block; font-family:var(--mono); font-weight:500;
  font-size:1.6rem; font-variant-numeric:tabular-nums; line-height:1.2; }}
.stat span {{ font-size:.78rem; color:var(--ink3); letter-spacing:.04em }}
.disclaimer {{ margin-block:28px 0; padding:16px 20px;
  border-inline-start:3px solid var(--gold); background:var(--card);
  color:var(--ink2); font-size:.88rem; max-inline-size:72ch; }}

nav {{ position:sticky; top:20px; max-block-size:calc(100vh - 40px);
  overflow-y:auto; padding-block:8px; }}
@media (max-width:900px) {{ nav {{ position:static; max-block-size:none;
  display:flex; flex-wrap:wrap; gap:6px; margin-block:24px 40px; }} }}
nav a {{ display:flex; align-items:baseline; gap:9px;
  padding:5px 9px; border-radius:7px; color:var(--ink2);
  text-decoration:none; font-size:.86rem; }}
nav a:hover {{ background:var(--card); color:var(--ink) }}
nav a:focus-visible {{ outline:2px solid var(--melody); outline-offset:1px }}
.nav-sign {{ font-family:var(--bible); font-size:1.15rem; color:var(--ink);
  min-inline-size:1.6em; }}
.nav-name {{ flex:1 }}
.nav-n {{ font-family:var(--mono); font-size:.72rem; color:var(--ink3);
  font-variant-numeric:tabular-nums }}
@media (max-width:900px) {{ .nav-n {{ display:none }} }}

.card {{ background:var(--card); border:1px solid var(--rule);
  border-radius:13px; padding:26px 28px 24px; margin-block-end:20px;
  scroll-margin-top:20px; }}
.card-head {{ display:flex; gap:22px; align-items:flex-start }}
.glyph {{ font-family:var(--bible); font-size:3.4rem; line-height:1;
  color:var(--ink); flex:none; padding-block-start:2px;
  min-inline-size:1.5em; text-align:center; }}
.titles h2 {{ font-family:var(--serif); font-weight:700; font-size:1.85rem;
  margin:0; line-height:1.15; }}
.seph {{ margin:.2em 0 0; font-size:.88rem; color:var(--ink3) }}
.facts {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px;
  margin:.7em 0 0 }}
.chip {{ font-size:.75rem; font-weight:500; padding:3px 10px;
  border-radius:999px; border:1px solid var(--rule); }}
.role-dis {{ color:var(--ink); border-color:var(--ink3) }}
.role-con {{ color:var(--ink2) }}
.sev-strict {{ color:var(--clay); border-color:currentColor }}
.sev-moderate {{ color:var(--gold); border-color:currentColor }}
.sev-lenient {{ color:var(--ink3) }}
.count {{ font-family:var(--mono); font-size:.76rem; color:var(--ink3);
  font-variant-numeric:tabular-nums }}
.role-he {{ margin:16px 0 0; color:var(--ink2); max-inline-size:68ch }}

.pos {{ display:flex; align-items:center; gap:12px; margin-block-start:16px }}
.pos-track {{ position:relative; block-size:3px; flex:1; max-inline-size:230px;
  background:var(--rule); border-radius:2px }}
.pos-track i {{ position:absolute; top:-3px; inline-size:9px; block-size:9px;
  border-radius:50%; background:var(--melody); transform:translateX(50%) }}
.pos-t {{ font-size:.78rem; color:var(--ink3) }}

.split {{ display:grid; grid-template-columns:1fr 1fr; gap:26px;
  margin-block-start:22px; padding-block-start:20px;
  border-block-start:1px solid var(--rule); }}
@media (max-width:640px) {{ .split {{ grid-template-columns:1fr; gap:22px }} }}
.split h3 {{ font-size:.73rem; font-weight:700; letter-spacing:.09em;
  color:var(--ink3); margin:0 0 12px; text-transform:none }}
.curve {{ inline-size:100%; block-size:auto; display:block;
  max-inline-size:320px }}
.curve .band {{ fill:var(--melody-soft); stroke:none }}
.curve .mean {{ fill:none; stroke:var(--melody); stroke-width:2;
  stroke-linecap:round; stroke-linejoin:round }}
.curve .zero {{ stroke:var(--rule); stroke-width:1; stroke-dasharray:3 4 }}
.contour {{ margin:10px 0 0; font-size:.9rem; color:var(--ink2) }}
.meta-line {{ margin:5px 0 0; font-family:var(--mono); font-size:.74rem;
  color:var(--ink3); font-variant-numeric:tabular-nums }}
.unchecked {{ margin:0; font-size:.9rem; color:var(--ink3);
  border-inline-start:2px solid var(--rule); padding-inline-start:12px }}

.nb {{ display:flex; align-items:center; gap:10px; margin-block-end:9px }}
.nb-l {{ font-size:.85rem; color:var(--ink2); min-inline-size:11em }}
.nb-bar {{ flex:1; block-size:6px; background:var(--rule); border-radius:3px;
  overflow:hidden }}
.nb-bar i {{ display:block; block-size:100%; background:var(--melody) }}
.nb-p {{ font-family:var(--mono); font-size:.76rem; color:var(--ink2);
  font-variant-numeric:tabular-nums; min-inline-size:2.6em; text-align:start }}

footer {{ max-inline-size:1180px; margin-inline:auto;
  padding:40px 28px 80px; border-block-start:1px solid var(--rule);
  color:var(--ink3); font-size:.85rem; max-inline-size:78ch }}
footer code {{ font-family:var(--mono); font-size:.9em; color:var(--ink2) }}
</style>

<header class="top">
  <h1>תיקון סופרים</h1>
  <p class="sub">כל טעמי המקרא בנוסח ספרדי-ירושלמי. כל מספר כאן נמדד —
    השכיחות, המיקום והתחביר נספרו מהטקסט המוטעם של התורה כולה; צורות
    המנגינה מוצעו מהקלטות אמיתיות. שום דבר לא הועתק מטבלה.</p>
  <div class="stats">
    <div class="stat"><b>{total:,}</b><span>מילים מוטעמות</span></div>
    <div class="stat"><b>{len(d)}</b><span>טעמים</span></div>
    <div class="stat"><b>{len(checkable)}</b><span>ניתנים לבדיקה אקוסטית</span></div>
    <div class="stat"><b>{len(d)-len(checkable)}</b><span>נדירים מכדי לבדוק</span></div>
  </div>
  <p class="disclaimer">המנגינות הן של בעל הקריאה שהוקלט. מסורות אחרות שרות
    אחרת, וגם בתוך אותה מסורת יש הבדלים בין בעלי קריאה. דירוג החומרה מסכם
    מקורות נגישים כדי לתעדף משוב באפליקציה — <strong>ואינו פסק הלכה</strong>.</p>
</header>

<div class="wrap">
  <nav aria-label="ניווט בטעמים">{nav}</nav>
  <main>{''.join(cards)}</main>
</div>

<footer>
  נוצר על ידי <code>Data_Prep/build_tikkun.py</code> ו־<code>build_tikkun_page.py</code>.
  השכיחויות, המיקומים והתחביר נספרים מהטקסט המוטעם; עקומות המנגינה ממוצעות
  מהקלטות לאחר נרמול לגובה הממוצע של המילה עצמה, כך שהן מתארות
  <strong>צורה</strong> ולא גובה מוחלט. הרצועה סביב הקו היא הפיזור בין
  המופעים — ברוב הטעמים היא רחבה מהצורה עצמה, כלומר העקומה היא נטייה ולא
  תבנית. כשייאספו קולות נוספים, הרצה חוזרת מעדכנת את כל המספרים.
</footer>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"  {len(d)} כרטיסים · {len(checkable)} עם עקומה")
    print(f"  נשמר: {OUT}")


if __name__ == "__main__":
    main()
