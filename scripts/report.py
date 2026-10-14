#!/usr/bin/env python3
"""
Build the HTML report page for a folder of API results (scores/ or supp/).

    scripts/report.py            # reads scores/results.jsonl, writes scores/report.html
    scripts/report.py supp       # another results folder

The page has a per-book summary table and one card per scored page: thumbnail,
the three probabilities, the caption, and the Qwen verdict when one was given.
Filter buttons narrow the grid to what needs a human look: high scores, pages
where the model and the caption disagree, the uncertain band, and pages where
Qwen and the API disagree.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MAO_RE, PAGES, ROOT, THRESH, load_ocr, load_results, qwen_says_yes  # noqa: E402


def pill_class(p):
    if p is None:
        return "na"
    return "hi" if p >= 0.8 else "mid" if p >= THRESH else "low" if p >= 0.2 else "no"


def fmt(p):
    return "–" if p is None else f"{p:.2f}"


def build(out: Path, qwen: dict | None = None):
    """qwen: optional {(book, page): qwen row} to show its verdict on each card."""
    rows = load_results(out / "results.jsonl")
    qwen = qwen or {}
    img_prefix = os.path.relpath(PAGES, out) + "/"
    books = sorted({b for b, _ in rows})
    ocr = {b: load_ocr(b) for b in books}

    def score(r):
        return max(r.get("p_character") or 0, r.get("p_image") or 0, r.get("p_any") or 0)

    # ---- per-book summary
    summary = []
    for b in books:
        rs = [r for (bb, _), r in rows.items() if bb == b]
        summary.append((
            b,
            (ocr[b].get("00001.jpg") or ocr[b].get("00003.jpg") or "").replace("\n", " ")[:40],
            len(rs),
            sum(1 for r in rs if MAO_RE.search(ocr[b].get(r["page"], ""))),
            sum(1 for r in rs if (r.get("p_character") or 0) >= THRESH),
            sum(1 for r in rs if (r.get("p_image") or 0) >= THRESH),
            sum(1 for r in rs if score(r) >= THRESH),
            sum(1 for r in rs if any(0.2 <= (r.get(f) or 0) < 0.8 for f in ("p_character", "p_image"))),
            sum(1 for r in rs if r.get("error") or r.get("refusal")),
        ))

    # ---- page cards
    cards = []
    for (b, p), r in sorted(rows.items()):
        cap = ocr[b].get(p, "")
        named = bool(MAO_RE.search(cap))
        pc, pi, pa = r.get("p_character"), r.get("p_image"), r.get("p_any")
        hi_c = (pc or 0) >= THRESH
        hi_i = (pi or 0) >= THRESH or ((pa or 0) >= THRESH and not hi_c)
        api_yes = score(r) >= THRESH
        err = r.get("error") or ("refusal" if r.get("refusal") else "")

        flags = ["all"]
        if hi_c: flags.append("char")
        if hi_i: flags.append("img")
        if named and not api_yes: flags.append("named-low")
        if api_yes and not named: flags.append("unnamed-high")
        if any(0.2 <= (x or 0) < 0.8 for x in (pc, pi)): flags.append("uncertain")
        if err: flags.append("error")

        chips = ""
        if pa is not None:
            chips += f' <span class="pill {pill_class(pa)}" title="any form">任 {fmt(pa)}</span>'
        if named:
            chips += ' <span class="tag">caption names Mao</span>'
        q = qwen.get((b, p))
        if q is not None:
            qy = qwen_says_yes(q)
            if qy != api_yes: flags.append("disagree")
            if qy and not api_yes: flags.append("qwen-only")
            if api_yes and not qy: flags.append("api-only")
            chips += (f' <span class="qwen" title="{html.escape(str(q.get("reason", ""))[:300])}">'
                      f'Qwen 人:{q.get("mao_person", "?")} 像:{q.get("mao_image", "?")}</span>')
        if r.get("tier"):
            chips += f' <span class="tag">tier {r["tier"]}</span>'
        if err:
            chips += f' <span class="err">{html.escape(err[:80])}</span>'

        src = f"{img_prefix}{html.escape(b)}/{p}"
        cards.append(
            f'<div class="card" data-book="{html.escape(b)}" data-f="{" ".join(flags)}" '
            f'data-pc="{-1 if pc is None else pc}" data-pi="{-1 if pi is None else pi}" data-pa="{-1 if pa is None else pa}">'
            f'<a href="{src}" target="_blank"><img loading="lazy" src="{src}"></a>'
            f'<div class="meta"><span class="pg">{p[:-4]}</span> '
            f'<span class="pill {pill_class(pc)}" title="Mao drawn as character">人 {fmt(pc)}</span> '
            f'<span class="pill {pill_class(pi)}" title="image of Mao in scene">像 {fmt(pi)}</span>'
            f'{chips}</div>'
            f'<div class="cap">{html.escape(cap[:160]) or "<i>no text</i>"}</div></div>'
        )

    summary_rows = "".join(
        f"<tr><td><code>{html.escape(b)}</code><br><small>{html.escape(t)}</small></td>"
        f"<td>{n}</td><td>{named}</td><td>{ch}</td><td>{im}</td><td>{either}</td><td>{unc}</td><td>{err}</td></tr>"
        for b, t, n, named, ch, im, either, unc, err in summary
    )
    book_opts = "".join(f'<option value="{html.escape(b)}">{html.escape(b)}</option>' for b in books)
    qwen_buttons = (
        '<button data-f="disagree">Qwen ≠ API</button>'
        '<button data-f="qwen-only">Qwen yes, API no</button>'
        '<button data-f="api-only">API yes, Qwen no</button>'
    ) if qwen else ""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Mao review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fff; --fg:#222; --mut:#666; --line:#ddd; --card:#fafafa;
  --hi:#c62828; --mid:#ef6c00; --low:#9e9d24; --no:#9e9e9e; --na:#bbb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141414; --fg:#eee; --mut:#aaa; --line:#333; --card:#1e1e1e; --no:#666; --na:#555; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
table {{ border-collapse:collapse; margin:8px 0 16px; }} th,td {{ border:1px solid var(--line); padding:4px 8px; text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); display:flex; gap:6px; flex-wrap:wrap; align-items:center; z-index:1; }}
button, select {{ font:inherit; padding:4px 10px; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:4px; cursor:pointer; }}
button.on {{ background:var(--fg); color:var(--bg); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:10px; margin-top:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.card img {{ width:100%; display:block; aspect-ratio:1000/1180; object-fit:contain; background:#000; }}
.meta {{ padding:6px 8px 0; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }}
.pg {{ font-weight:600; }} .pill {{ border-radius:10px; padding:1px 8px; color:#fff; font-size:12px; }}
.hi {{ background:var(--hi); }} .mid {{ background:var(--mid); }} .low {{ background:var(--low); }} .no {{ background:var(--no); }} .na {{ background:var(--na); }}
.tag {{ font-size:11px; color:var(--mut); border:1px dashed var(--mut); border-radius:4px; padding:0 4px; }}
.qwen {{ font-size:11px; border:1px solid var(--mut); border-radius:4px; padding:0 4px; cursor:help; }}
.err {{ font-size:11px; color:var(--hi); }}
.cap {{ padding:4px 8px 8px; font-size:12px; color:var(--mut); }}
.count {{ color:var(--mut); margin-left:auto; }}
.hidden {{ display:none; }}
</style></head><body>
<h1>Mao review: {len(rows)} pages in {len(books)} books</h1>
<p><b>人</b> = probability Mao is drawn as a character, <b>像</b> = probability an image of Mao
(portrait, poster, badge, statue, face in a sun) is in the scene, <b>任</b> = catch-all probability
that Mao appears in any form. Counts below use a threshold of {THRESH}. Click a thumbnail for the full page.</p>
<table><tr><th>book</th><th>pages</th><th>caption names Mao</th><th>人 ≥ {THRESH}</th><th>像 ≥ {THRESH}</th><th>any ≥ {THRESH}</th><th>uncertain (0.2–0.8)</th><th>errors</th></tr>
{summary_rows}</table>
<div class="bar">
<select id="book"><option value="">all books</option>{book_opts}</select>
<button data-f="all" class="on">all pages</button>
<button data-f="char">人 ≥ {THRESH}</button>
<button data-f="img">像 ≥ {THRESH}</button>
<button data-f="unnamed-high">high but caption silent</button>
<button data-f="named-low">caption names Mao but low</button>
<button data-f="uncertain">uncertain</button>
<button data-f="error">errors</button>
{qwen_buttons}
<select id="sort"><option value="page">sort by page</option><option value="pc">sort by 人</option><option value="pi">sort by 像</option><option value="pa">sort by 任</option></select>
<span class="count" id="count"></span>
</div>
<div class="grid" id="grid">{"".join(cards)}</div>
<script>
const grid=document.getElementById('grid'), cards=[...grid.children];
let flag='all', book='', sortk='page';
function apply(){{
  let n=0;
  for(const c of cards){{
    const ok=c.dataset.f.split(' ').includes(flag) && (!book || c.dataset.book===book);
    c.classList.toggle('hidden',!ok); if(ok) n++;
  }}
  const vis=cards.slice();
  if(sortk!=='page') vis.sort((a,b)=>parseFloat(b.dataset[sortk])-parseFloat(a.dataset[sortk]));
  vis.forEach(c=>grid.appendChild(c));
  document.getElementById('count').textContent=n+' shown';
}}
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.bar button').forEach(x=>x.classList.remove('on')); b.classList.add('on'); flag=b.dataset.f; apply();}});
document.getElementById('book').onchange=e=>{{book=e.target.value; apply();}};
document.getElementById('sort').onchange=e=>{{sortk=e.target.value; apply();}};
apply();
</script></body></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")
    print(f"wrote {out / 'report.html'}\n")
    print(f"{'book':34} {'pages':>5} {'named':>5} {'人≥.5':>5} {'像≥.5':>5} {'any':>5} {'unc':>4} {'err':>4}")
    for b, _, n, named, ch, im, either, unc, err in summary:
        print(f"{b:34} {n:5} {named:5} {ch:5} {im:5} {either:5} {unc:4} {err:4}")


if __name__ == "__main__":
    build(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "scores"))
