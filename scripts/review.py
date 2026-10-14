#!/usr/bin/env python3
"""
Stage 4: build the visual review page, package it for a collaborator, and turn the
reviewer's decisions into the final table.

    scripts/review.py                     # supp/review.html
    scripts/review.py --package           # review_package.zip: page + its images + README
    scripts/review.py --finalize          # supp/review.csv + scores -> mao_pages.csv

The page has two groups of pages:

    uncertain   Qwen and the API disagree, an API score is in the 0.2-0.8 band,
                or Qwen said unsure. These need a decision.
    agreed      both systems say Mao is present. The answer is pre-filled from
                the scores and can be corrected.

Images are linked from lhh/, never embedded. Keys on the page:

    1 = Mao drawn as a character    2 = image of Mao in the scene
    3 = both                        0 = neither        u = clear
    j / k or arrow keys move, Enter opens the full page

Decisions are kept in the browser (localStorage) and exported with
"Export CSV" as review.csv (book, page, decision, group). Save that file into
the supp folder and run --finalize to produce mao_pages.csv.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import ROOT, THRESH, load_qwen, qwen_says_yes  # noqa: E402

DECISIONS = ("character", "image", "both", "none")

README = """HOW TO REVIEW THE PAGES

1. Unzip this file. Keep everything inside the unzipped folder together.
2. Double-click  review.html  to open it in your web browser (Chrome, Safari,
   Firefox or Edge all work). It opens from your computer; no internet needed.
3. The pages come in two groups, chosen with the buttons at the top:

   UNCERTAIN  - two computer programs disagreed about these. Each needs your
                decision. This is the main work.
   AGREED     - both programs say Mao is in the picture, and an answer is
                already filled in. Browse these to catch mistakes and change
                the answer where it is wrong.

4. Each card shows one page of a picture book. Decide whether Mao Zedong is
   in the drawing, and in which way. Press one key per page:

       1  Mao is drawn as a person in the scene (any age, even a small child)
       2  an image of Mao is in the scene: a portrait on a wall, a poster,
          a statue, a badge, a banner, a book cover, a face in a sun
       3  both of the above
       0  neither: Mao is not in the picture at all
       u  clear your answer

   The page jumps to the next card after each answer. Use the j and k keys
   or the arrow keys to move without answering. Press Enter to open the
   page at full size. Click "undecided" at the top to see what is left.
5. Your answers are saved automatically in the browser. You can close the
   window and continue later on the same computer and browser.
6. When you are finished, click "Export CSV" at the top. A file named
   review.csv is downloaded. Send that file back.

Hints: the blue text on each card says why the page is there. The "Qwen" and
"API" lines are the two programs' guesses; they are often wrong, so trust your
eyes. Small portraits on walls, on stage backdrops or on banners are the most
common thing to look for. A quotation board or slogan with Mao's name but no
face does not count.
"""


def api_verdict(r: dict) -> tuple[bool, bool, bool]:
    """(character, image, uncertain) from the API scores."""
    pc, pi, pa = (r.get(f) or 0 for f in C.SCORE_FIELDS)
    ch = pc >= THRESH
    im = pi >= THRESH or (pa >= THRESH and not ch)
    unc = any(0.2 <= x < 0.8 for x in (pc, pi, pa))
    return ch, im, unc


def default_decision(ch: bool, im: bool) -> str:
    return "both" if ch and im else "character" if ch else "image" if im else "none"


def classify(rows: dict, qwen: dict) -> dict:
    """(book, page) -> 'uncertain' | 'agreed' | 'rejected' for every scored page."""
    groups = {}
    for k, r in rows.items():
        if r.get("p_character") is None:
            continue
        q = qwen.get(k, {})
        ch, im, unc = api_verdict(r)
        ay, qy = ch or im, qwen_says_yes(q)
        unsure = "unsure" in (q.get("mao_person"), q.get("mao_image"))
        if ay != qy or unc or unsure:
            groups[k] = "uncertain"
        elif ay and qy:
            groups[k] = "agreed"
        else:
            groups[k] = "rejected"
    return groups


def build(out: Path, qdir: Path, img_prefix: str | None = None, page_path: Path | None = None) -> list:
    """Write the review page; returns the (book, page) keys it contains."""
    if img_prefix is None:
        img_prefix = os.path.relpath(C.PAGES, out) + "/"
    rows = C.load_results(out / "results.jsonl")
    qwen = load_qwen(qdir)
    groups = classify(rows, qwen)
    keys = sorted(k for k, g in groups.items() if g != "rejected")
    n_unc = sum(1 for k in keys if groups[k] == "uncertain")
    books = sorted({b for b, _ in keys})
    ocr = {b: C.load_ocr(b) for b in books}
    print(f"{len(keys)} pages on the review page: {n_unc} uncertain, {len(keys) - n_unc} agreed, in {len(books)} books")

    cards = []
    for i, (b, p) in enumerate(keys):
        r, q = rows[(b, p)], qwen.get((b, p), {})
        ch, im, unc = api_verdict(r)
        group = groups[(b, p)]
        if group == "agreed":
            why, default = "agreed, pre-filled", default_decision(ch, im)
        else:
            why, default = " · ".join(
                w for w, c in (("Qwen ≠ API", (ch or im) != qwen_says_yes(q)), ("API uncertain", unc),
                               ("Qwen unsure", "unsure" in (q.get("mao_person"), q.get("mao_image")))) if c), ""
        src = f"{img_prefix}{html.escape(b)}/{p}"
        cap = ocr[b].get(p, "")
        scores = " ".join(f"{n} {'–' if r.get(f) is None else f'{r[f]:.2f}'}"
                          for n, f in zip(("人", "像", "任"), C.SCORE_FIELDS))
        cards.append(
            f'<div class="card {group}" data-key="{html.escape(b)}/{p}" data-book="{html.escape(b)}" '
            f'data-group="{group}" data-default="{default}">'
            f'<div class="head"><b>{i + 1}.</b> <code>{html.escape(b)}</code> page {p[:-4]}'
            f' <span class="why {group}">{why}</span>'
            f' <a href="{src}" target="_blank">open</a></div>'
            f'<a href="{src}" target="_blank"><img loading="lazy" src="{src}"></a>'
            f'<div class="info"><div><b>API</b> {scores}' + (f' · tier {r["tier"]}' if r.get("tier") else "") + '</div>'
            f'<div><b>Qwen</b> 人:{q.get("mao_person", "–")} 像:{q.get("mao_image", "–")}'
            f' <span class="reason">{html.escape(str(q.get("reason", ""))[:220])}</span></div>'
            f'<div class="cap">{html.escape(cap[:200]) or "<i>no text</i>"}</div>'
            f'<div class="dec"><button data-d="character">1 人 drawn</button><button data-d="image">2 像 image</button>'
            f'<button data-d="both">3 both</button><button data-d="none">0 neither</button>'
            f'<button data-d="">u clear</button><span class="chosen"></span></div></div></div>'
        )
    book_opts = "".join(f'<option value="{html.escape(b)}">{html.escape(b)}</option>' for b in books)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Mao page review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fff; --fg:#222; --mut:#666; --line:#ddd; --card:#fafafa; --acc:#1565c0; --sel:#fff3e0; --ok:#2e7d32; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141414; --fg:#eee; --mut:#aaa; --line:#333; --card:#1e1e1e; --acc:#64b5f6; --sel:#3a2e1a; --ok:#81c784; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }}
h1 {{ font-size:20px; margin:0 0 6px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); display:flex; gap:8px; flex-wrap:wrap; align-items:center; z-index:2; }}
button, select {{ font:inherit; padding:4px 10px; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:4px; cursor:pointer; }}
button.on {{ background:var(--fg); color:var(--bg); }}
.sep {{ width:1px; height:22px; background:var(--line); }}
.card {{ max-width:1000px; margin:14px auto; background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px; }}
.card.cur {{ outline:3px solid var(--acc); }}
.card.done {{ background:var(--sel); }}
.card img {{ width:100%; display:block; background:#000; }}
.head {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; margin-bottom:6px; }}
.why {{ font-weight:600; color:var(--acc); }} .why.agreed {{ color:var(--ok); }}
.info {{ margin-top:6px; display:flex; flex-direction:column; gap:3px; }}
.reason, .cap {{ color:var(--mut); font-size:12px; }}
.dec {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-top:4px; }}
.chosen {{ font-weight:600; color:var(--acc); }}
.hidden {{ display:none; }}
.help {{ color:var(--mut); font-size:12px; }}
</style></head><body>
<h1>Mao page review: {n_unc} uncertain pages to decide, {len(keys) - n_unc} agreed pages to check</h1>
<div class="help">Keys: <b>1</b> drawn · <b>2</b> image · <b>3</b> both · <b>0</b> neither · <b>u</b> clear · <b>j/k</b> or arrows move · <b>Enter</b> opens the page.
Decisions stay in this browser; export them when done.</div>
<div class="bar">
<button data-g="uncertain" class="on">uncertain</button><button data-g="agreed">agreed</button><button data-g="">both groups</button>
<span class="sep"></span>
<button data-f="all" class="on">all</button><button data-f="todo">undecided</button><button data-f="done">decided</button>
<span class="sep"></span>
<select id="book"><option value="">all books</option>{book_opts}</select>
<button id="export">Export CSV</button><label><button onclick="document.getElementById('imp').click()">Import CSV</button><input id="imp" type="file" accept=".csv" style="display:none"></label>
<span id="count"></span>
</div>
<div id="list">{"".join(cards)}</div>
<script>
const KEY='mao-review:{html.escape(out.name)}';
let dec={{}}; try{{dec=JSON.parse(localStorage.getItem(KEY)||'{{}}');}}catch(e){{}}
function save(){{ try{{localStorage.setItem(KEY,JSON.stringify(dec));}}catch(e){{}} }}
const cards=[...document.querySelectorAll('.card')];
let cur=0, flt='all', grp='uncertain', book='';
function val(c){{ return c.dataset.key in dec ? dec[c.dataset.key] : (c.dataset.default||''); }}
function paint(c){{
  const d=val(c), own=c.dataset.key in dec;
  c.classList.toggle('done',!!d);
  c.querySelector('.chosen').textContent=d?('→ '+d+(own?'':' (pre-filled)')):'';
  c.querySelectorAll('.dec button').forEach(b=>b.classList.toggle('on',b.dataset.d===d&&d!==''));
}}
function visible(){{ return cards.filter(c=>!c.classList.contains('hidden')); }}
function apply(){{
  for(const c of cards){{
    const d=!!val(c);
    const ok=(!grp||c.dataset.group===grp)&&(!book||c.dataset.book===book)&&(flt==='all'||(flt==='todo'&&!d)||(flt==='done'&&d));
    c.classList.toggle('hidden',!ok);
  }}
  const unc=cards.filter(c=>c.dataset.group==='uncertain'), agr=cards.filter(c=>c.dataset.group==='agreed');
  const uDone=unc.filter(c=>val(c)).length, aChanged=agr.filter(c=>c.dataset.key in dec).length;
  document.getElementById('count').textContent=`${{visible().length}} shown · uncertain ${{uDone}} / ${{unc.length}} decided · agreed ${{aChanged}} / ${{agr.length}} changed`;
  focus(Math.min(cur,Math.max(0,visible().length-1)),false);
}}
function focus(i,scroll=true){{
  const v=visible(); if(!v.length) return; cur=Math.max(0,Math.min(i,v.length-1));
  cards.forEach(c=>c.classList.remove('cur')); v[cur].classList.add('cur');
  if(scroll) v[cur].scrollIntoView({{block:'center',behavior:'smooth'}});
}}
function decide(c,d){{ if(d) dec[c.dataset.key]=d; else delete dec[c.dataset.key]; save(); paint(c); apply(); }}
cards.forEach(c=>{{ paint(c);
  c.addEventListener('click',e=>{{ if(e.target.tagName!=='A'&&e.target.tagName!=='IMG') focus(visible().indexOf(c),false); }});
  c.querySelectorAll('.dec button').forEach(b=>b.onclick=e=>{{e.stopPropagation(); decide(c,b.dataset.d);}});
}});
document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  const v=visible(); const c=v[cur]; if(!c) return;
  const m={{'1':'character','2':'image','3':'both','0':'none','u':''}};
  if(e.key in m){{ decide(c,m[e.key]); if(e.key!=='u'&&flt!=='todo') focus(cur+1); e.preventDefault(); }}
  else if(e.key==='j'||e.key==='ArrowDown'||e.key==='ArrowRight'){{ focus(cur+1); e.preventDefault(); }}
  else if(e.key==='k'||e.key==='ArrowUp'||e.key==='ArrowLeft'){{ focus(cur-1); e.preventDefault(); }}
  else if(e.key==='Enter'){{ window.open(c.querySelector('img').src,'_blank'); }}
}});
document.querySelectorAll('.bar button[data-g]').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.bar button[data-g]').forEach(x=>x.classList.remove('on')); b.classList.add('on'); grp=b.dataset.g; cur=0; apply(); }});
document.querySelectorAll('.bar button[data-f]').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.bar button[data-f]').forEach(x=>x.classList.remove('on')); b.classList.add('on'); flt=b.dataset.f; cur=0; apply(); }});
document.getElementById('book').onchange=e=>{{book=e.target.value; cur=0; apply();}};
document.getElementById('export').onclick=()=>{{
  let s='book,page,decision,group\\n';
  for(const c of cards){{ const d=val(c); if(d){{ const [b,p]=c.dataset.key.split('/'); s+=`${{b}},${{p}},${{d}},${{c.dataset.group}}\\n`; }} }}
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}})); a.download='review.csv'; a.click();
}};
document.getElementById('imp').onchange=e=>{{
  const f=e.target.files[0]; if(!f) return; const rd=new FileReader();
  rd.onload=()=>{{ for(const line of rd.result.split(/\\r?\\n/).slice(1)){{ const [b,p,d]=line.split(','); if(b&&p&&d) dec[b+'/'+p]=d.trim(); }}
    save(); cards.forEach(paint); apply(); }};
  rd.readAsText(f);
}};
apply();
</script></body></html>"""
    page_path = page_path or out / "review.html"
    page_path.write_text(page, encoding="utf-8")
    print(f"wrote {page_path}")
    return keys


def package(out: Path, qdir: Path, name: str = "review_package"):
    """Self-contained folder + zip: review.html, the images it needs, a README."""
    pkg = ROOT / name
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "lhh").mkdir(parents=True)
    keys = build(out, qdir, img_prefix="lhh/", page_path=pkg / "review.html")
    size = 0
    for b, p in keys:
        dst = pkg / "lhh" / b / p
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(C.PAGES / b / p, dst)
        size += dst.stat().st_size
    (pkg / "README.txt").write_text(README, encoding="utf-8")
    zpath = ROOT / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    print(f"{len(keys)} images, {size / 1e6:.0f} MB; zip: {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


def finalize(out: Path, qdir: Path):
    rows = C.load_results(out / "results.jsonl")
    qwen = load_qwen(qdir)
    decided = {}
    rf = out / "review.csv"
    if rf.exists():
        for r in csv.DictReader(rf.open(encoding="utf-8")):
            decided[(r["book"], r["page"])] = r["decision"].strip()
        print(f"{len(decided)} reviewer decisions from {rf}")
    else:
        print(f"no {rf}: using the scores alone")

    final, unknown = [], 0
    for k in sorted(set(rows) | set(decided)):
        r = rows.get(k, {})
        if k in decided:
            d = decided[k]
            if d not in DECISIONS:
                unknown += 1
                continue
            ch, im, src = d in ("character", "both"), d in ("image", "both"), "reviewer"
        elif r.get("p_character") is not None:
            ch, im, _ = api_verdict(r)
            src = "api+qwen" if qwen_says_yes(qwen.get(k, {})) else "api"
        else:
            continue
        if ch or im:
            final.append((*k, ch, im, src))

    with (ROOT / "mao_pages.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["book", "page", "character", "image", "source"])
        for b, p, ch, im, src in final:
            w.writerow([b, p, "yes" if ch else "no", "yes" if im else "no", src])
    n_ch = sum(1 for f in final if f[2])
    n_im = sum(1 for f in final if f[3])
    print(f"wrote {ROOT / 'mao_pages.csv'}: {len(final)} pages in {len({f[0] for f in final})} books "
          f"({n_ch} with Mao drawn, {n_im} with an image of Mao)")
    if unknown:
        print(f"  {unknown} review rows had an unknown decision and were skipped")


def main():
    ap = argparse.ArgumentParser(description="visual review page, collaborator package, final table")
    ap.add_argument("--out", default="supp", help="results folder (default supp)")
    ap.add_argument("--qwen", default="qwen", help="folder with Qwen jsonl files")
    ap.add_argument("--package", action="store_true", help="zip the page with its images for a collaborator")
    ap.add_argument("--finalize", action="store_true", help="build mao_pages.csv from review.csv and the scores")
    args = ap.parse_args()
    out, qdir = ROOT / args.out, ROOT / args.qwen
    if args.finalize:
        finalize(out, qdir)
    elif args.package:
        package(out, qdir)
    else:
        build(out, qdir)


if __name__ == "__main__":
    main()
