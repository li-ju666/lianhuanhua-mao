#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["httpx>=0.27"]
# ///
"""
Stage 3b (optional): score whole books, or the whole corpus, with the OpenAI Decisions API.
The targeted run over the hard pages is scripts/supplement.py; this is the full scorer.

Each page is sent as an image (1000 px wide, as stored in lhh/) together with
text context built from the OCR: the book's title line (plus its synopsis when
the book names Mao anywhere), the two previous captions and the page's own
caption. The model answers three predicate questions with probabilities:

    mao_character   Mao drawn as a person in the panel, at any age
    mao_image       an image of Mao inside the scene: portrait, poster, statue,
                    badge, banner, book cover, face in a sun
    mao_any         catch-all: Mao's likeness in any form (recall safety net)

Key: OPENAI_API_KEY in the environment, or a file openai.key at the repository root.
Needs httpx (uv run scripts/api_score.py, or pip install httpx).

Usage:
    scripts/api_score.py BOOK [BOOK ...]        # given book folders
    scripts/api_score.py --all                  # every book in lhh/
    scripts/api_score.py --out scores --jobs 16 # options (these are the defaults)
    scripts/api_score.py --limit 20 BOOK        # only the first 20 pages of each book

Results go to <out>/results.jsonl (one line per page, resumable: scored pages
are skipped on re-run) and <out>/results.csv. The HTML review page
<out>/report.html is built at the end; scripts/report.py rebuilds it on demand.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import PAGES, SCORE_FIELDS  # noqa: E402

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
PRICE_PER_M = 0.10  # USD per million input tokens

QUESTIONS = [
    {
        "type": "predicate",
        "name": "mao_character",
        "instructions": (
            "Is Mao Zedong (毛泽东/毛主席) drawn as a living person in the panel, at any age, "
            "even small, in the background or in a montage? Use the context to identify "
            "unnamed figures (他 in a biography). A caption mention alone is not enough. "
            "Portraits, posters, statues, badges or banners showing him do not count here. "
            "Other people (Marx, Lenin, Sun Yat-sen, Zhou Enlai, Lin Biao, his parents, "
            "teachers, friends) do not count."
        ),
    },
    {
        "type": "predicate",
        "name": "mao_image",
        "instructions": (
            "Does the panel contain an image of Mao Zedong's face in any form other than a "
            "living participant: a framed or carried portrait, photograph, poster, painting, "
            "statue or bust, chest badge, banner or placard with his face, a book cover with "
            "his face, or a symbolic face in a sun or halo? Often small, on a wall behind the "
            "action, and not in the caption, so inspect the whole picture. Quotation boards, "
            "slogans, flags or text alone do not count. Portraits of other people (Marx, "
            "Lenin, Stalin, Sun Yat-sen, Lin Biao) do not count."
        ),
    },
    {
        "type": "predicate",
        "name": "mao_any",
        "instructions": (
            "Does Mao Zedong's likeness appear anywhere in the panel in any form: drawn as a "
            "person of any age, or as a portrait, poster, statue, badge, banner, book cover, "
            "or face in a sun? Use the context to identify unnamed figures, but a caption "
            "mention alone is not enough. Other leaders' faces do not count."
        ),
    },
]


def make_jobs(books: list[str], skip: set, limit: int = 0) -> list[tuple]:
    """(book, page, context) for every page of the books not in `skip`."""
    jobs = []
    for book in books:
        pages = C.book_pages(book)
        ocr = C.load_ocr(book)
        intro = C.book_intro(ocr, pages)
        for page in pages[:limit] if limit else pages:
            if (book, page) not in skip:
                jobs.append((book, page, C.context_text(book, page, pages, ocr, intro)))
    return jobs


# ----------------------------------------------------------------- scoring

def encode_image(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def score_page(client, key: str, book: str, page: str, text: str) -> dict:
    body = {
        "model": MODEL,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": encode_image(PAGES / book / page)},
        ]}],
        "questions": QUESTIONS,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    empty = {"book": book, "page": page, **{f: None for f in SCORE_FIELDS}}
    delay = 2.0
    err = "no attempt"
    for _ in range(8):
        try:
            r = client.post(URL, json=body, headers=headers, timeout=120)
        except Exception as e:
            err = f"network: {e}"
        else:
            if r.status_code == 200:
                data = r.json()
                probs, refusal = {}, None
                for a in data.get("answers", []):
                    if a.get("type") == "refusal":
                        refusal = a
                    elif "probability" in a:
                        probs[a["name"]] = a["probability"]
                return {
                    "book": book, "page": page,
                    "p_character": probs.get("mao_character"),
                    "p_image": probs.get("mao_image"),
                    "p_any": probs.get("mao_any"),
                    "refusal": refusal,
                    "usage": data.get("usage"),
                }
            if r.status_code in (400, 401, 403, 404, 422):
                return {**empty, "error": f"http {r.status_code}: {r.text[:500]}"}
            err = f"http {r.status_code}: {r.text[:200]}"
            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                if ra and ra.isdigit():
                    delay = max(delay, float(ra))
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return {**empty, "error": err}


def score_pages(jobs: list[tuple], jsonl: Path, key: str, workers: int, extra: dict | None = None):
    """Score (book, page, text) jobs in parallel, appending one line per page to jsonl.
    `extra` maps (book, page) -> fields to add to that page's result (e.g. tier)."""
    import httpx  # only the API calls need it
    lock = threading.Lock()
    n_ok = n_err = tokens = 0
    start = time.time()
    with httpx.Client() as client, jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(score_page, client, key, b, p, t) for b, p, t in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if extra:
                r.update(extra.get((r["book"], r["page"]), {}))
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                if r.get("error"):
                    n_err += 1
                    if n_err <= 5:
                        print(f"\n  error {r['book']}/{r['page']}: {r['error']}")
                else:
                    n_ok += 1
                tokens += (r.get("usage") or {}).get("input_tokens") or 0
                el = time.time() - start
                rate = i / el if el > 0 else 0
                eta = (len(jobs) - i) / rate / 60 if rate else 0
                print(f"\r  {i}/{len(jobs)}  ok {n_ok}  err {n_err}  {rate:.1f} pages/s  ~{eta:.0f} min left   ",
                      end="", flush=True)
    print(f"\ndone in {(time.time() - start) / 60:.1f} min; {n_ok} scored, {n_err} errors")
    if tokens:
        print(f"input tokens: {tokens:,}  (~${tokens / 1e6 * PRICE_PER_M:.2f}, {tokens / max(n_ok, 1):.0f} per page)")
    return n_ok, n_err, tokens


def write_csv(out: Path) -> dict:
    rows = C.load_results(out / "results.jsonl")
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["book", "page", *SCORE_FIELDS, "error"])
        for (b, p), r in sorted(rows.items()):
            w.writerow([b, p, *("" if r.get(f) is None else f"{r[f]:.3f}" for f in SCORE_FIELDS),
                        r.get("error", "") or ("refusal" if r.get("refusal") else "")])
    print(f"wrote {out / 'results.csv'} ({len(rows)} pages)")
    return rows


def build_report(out: Path, **kwargs):
    """Write results.csv and report.html; a report failure must not lose results."""
    write_csv(out)
    try:
        import report
        report.build(out, **kwargs)
    except Exception as e:
        print(f"report not built: {e}")


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="score lianhuanhua pages for Mao Zedong")
    ap.add_argument("books", nargs="*", help="book folder names under lhh/")
    ap.add_argument("--all", action="store_true", help="every book under lhh/")
    ap.add_argument("--jobs", type=int, default=16, help="parallel requests (default 16)")
    ap.add_argument("--out", default="scores", help="output folder (default scores)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages of each book")
    args = ap.parse_args()

    if args.all:
        books = sorted(p.name for p in PAGES.iterdir() if p.is_dir())
    elif args.books:
        books = args.books
    else:
        ap.error("give book folder names or --all")
    for b in books:
        if not (PAGES / b).is_dir():
            sys.exit(f"no such book folder: {PAGES / b}")

    key = C.load_key()
    if not key:
        sys.exit("no API key: set OPENAI_API_KEY or put the key in openai.key")

    out = C.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "results.jsonl"
    done = C.scored(C.load_results(jsonl))
    jobs = make_jobs(books, done, args.limit)
    print(f"{len(books)} books, {len(jobs)} pages to score, {len(done)} already done, {args.jobs} workers")
    if jobs:
        score_pages(jobs, jsonl, key, args.jobs)
    build_report(out)


if __name__ == "__main__":
    main()
