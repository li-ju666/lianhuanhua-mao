#!/usr/bin/env python3
"""Coverage and counts for every stage: which pages still lack OCR or a Qwen verdict,
how many pages are flagged, how many the API scored.  Run any time, on any node:

    scripts/status.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

books = C.books()
pages = {b: C.book_pages(b) for b in books}
n_pages = sum(len(v) for v in pages.values())
print(f"lhh:  {len(books)} books, {n_pages} pages")

ocr_rows = C.load_jsonl_dir(C.OCR)
miss = [(b, p) for b in books for p in pages[b] if (b, p) not in ocr_rows]
print(f"ocr:  {len(ocr_rows)} pages, missing {len(miss)}" + (f" in {len({b for b, _ in miss})} books" if miss else ""))

qwen = C.load_qwen()
miss = [(b, p) for b in books for p in pages[b] if (b, p) not in qwen]
print(f"qwen: {len(qwen)} pages, missing {len(miss)}" + (f" in {len({b for b, _ in miss})} books" if miss else ""))
if qwen:
    c = Counter()
    for q in qwen.values():
        c["person yes"] += q.get("mao_person") == "yes"
        c["image yes"] += q.get("mao_image") == "yes"
        c["unsure"] += "unsure" in (q.get("mao_person"), q.get("mao_image"))
        c["flagged"] += C.qwen_says_yes(q)
    flagged_books = len({b for (b, _), q in qwen.items() if C.qwen_says_yes(q)})
    print(f"      flagged {c['flagged']} pages in {flagged_books} books "
          f"(person yes {c['person yes']}, image yes {c['image yes']}, unsure {c['unsure']})")

api = C.load_results(C.SUPP / "results.jsonl")
if api:
    ok = C.scored(api)
    print(f"supp: {len(ok)} pages scored by the API, {len(api) - len(ok)} errors")
rv = C.SUPP / "review.csv"
if rv.exists():
    print(f"      reviewer decisions: {sum(1 for _ in open(rv, encoding='utf-8')) - 1}")
final = C.ROOT / "mao_pages.csv"
if final.exists():
    print(f"final: {sum(1 for _ in open(final, encoding='utf-8')) - 1} pages in {final.name}")
