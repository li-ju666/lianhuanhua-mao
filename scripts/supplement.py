#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["httpx>=0.27"]
# ///
"""
Stage 3: supplement the Qwen sweep with targeted OpenAI Decisions API scoring.
Needs httpx for the API calls (uv run scripts/supplement.py, or pip install httpx);
--dry-run works without it.

Reads qwen/*.jsonl (one line per page: mao_person, mao_image,
confidence, reason, book, page) and sends only the pages where Qwen is weak:

    tier 1  Qwen says yes or unsure on either column      -> remove its false positives
    tier 2  Qwen says no, but the caption names Mao        -> missed drawn Mao
    tier 3  Qwen says no, but a clause of its reason       -> it saw a portrait-like
            affirms a portrait-like object (画像/肖像/像章/塑像)  object and talked itself out
    tier 4  everything else in books that name Mao         -> only with --insurance
            somewhere (small wall portraits Qwen never flagged)

Usage:
    scripts/supplement.py --dry-run          # count pages per tier, no API calls
    scripts/supplement.py                    # score tiers 1-3 into supp/
    scripts/supplement.py --insurance        # also tier 4
    scripts/supplement.py --qwen DIR --out DIR --jobs N

Outputs in <out>/:
    results.jsonl   API scores for the pages sent (resumable; "tier" field added)
    results.csv     the same as a table
    merged.csv      every Qwen page with Qwen verdict, tier, API scores, final verdict
    report.html     review page for the API-scored pages, Qwen verdict on each card
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from api_score import PRICE_PER_M, build_report, score_pages  # noqa: E402
from common import ROOT, THRESH, load_qwen, qwen_says_yes  # noqa: E402

PORTRAIT_RE = re.compile("画像|肖像|像章|塑像|挂像|头像|半身像|画框|相框|雕像|胸像|画报|宣传画")
NEG_RE = re.compile("无|未|没有|不|非|缺")
EXAMPLE_RE = re.compile(r"^\s*(如|例如|包括|比如|诸如)")
EST_TOKENS_PER_PAGE = 2500


def affirms_portrait(reason: str) -> bool:
    """True when a clause of Qwen's reason mentions a portrait-like object without a
    negation and is not the boilerplate example list after 未见…(如画像、像章等)."""
    return any(
        PORTRAIT_RE.search(c) and not NEG_RE.search(c) and not EXAMPLE_RE.search(c)
        for c in re.split("[。；;，,：:（）()]", reason or "")
    )


def assign_tiers(qwen: dict, insurance: bool) -> tuple[dict, dict]:
    """Return ({(book, page): tier}, {book: ocr}); pages with no tier are not sent."""
    tiers, ocr_cache = {}, {}
    for (b, p), r in qwen.items():
        if b not in ocr_cache:
            ocr_cache[b] = C.load_ocr(b)
        ocr = ocr_cache[b]
        if qwen_says_yes(r):
            tiers[(b, p)] = 1
        elif C.names_mao(ocr.get(p, "")):
            tiers[(b, p)] = 2
        elif affirms_portrait(r.get("reason", "")):
            tiers[(b, p)] = 3
        elif insurance and any(C.names_mao(t) for t in ocr.values()):
            tiers[(b, p)] = 4
    return tiers, ocr_cache


def final_verdict(q: dict, a: dict | None) -> str:
    qy = qwen_says_yes(q)
    if a is None or a.get("p_character") is None:
        return "qwen_yes_unchecked" if qy else "qwen_no"
    ay = max(a.get(f) or 0 for f in C.SCORE_FIELDS) >= THRESH
    if qy and ay:
        return "both_yes"
    if not qy and not ay:
        return "both_no"
    return "api_yes_only" if ay else "qwen_yes_only"


def write_merged(out: Path, qwen: dict, tiers: dict, api: dict) -> dict:
    counts = {}
    with (out / "merged.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["book", "page", "qwen_person", "qwen_image", "qwen_conf", "tier", *C.SCORE_FIELDS, "verdict"])
        for k in sorted(qwen):
            q, a = qwen[k], api.get(k)
            v = final_verdict(q, a)
            counts[v] = counts.get(v, 0) + 1
            w.writerow([*k, q.get("mao_person", ""), q.get("mao_image", ""), q.get("confidence", ""),
                        tiers.get(k, ""),
                        *("" if a is None or a.get(f) is None else f"{a[f]:.3f}" for f in C.SCORE_FIELDS), v])
    print(f"wrote {out / 'merged.csv'} ({len(qwen)} pages)")
    for v in ("both_yes", "api_yes_only", "qwen_yes_only", "both_no", "qwen_yes_unchecked", "qwen_no"):
        if v in counts:
            print(f"  {v:20} {counts[v]:7}")
    return counts


def main():
    ap = argparse.ArgumentParser(description="targeted API scoring on top of Qwen results")
    ap.add_argument("--qwen", default="qwen", help="folder with Qwen jsonl files")
    ap.add_argument("--out", default="supp", help="output folder (default supp)")
    ap.add_argument("--jobs", type=int, default=16, help="parallel requests (default 16)")
    ap.add_argument("--insurance", action="store_true", help="also send tier 4")
    ap.add_argument("--dry-run", action="store_true", help="only count pages per tier")
    args = ap.parse_args()

    qwen = load_qwen(ROOT / args.qwen)
    if not qwen:
        sys.exit(f"no Qwen results found in {ROOT / args.qwen}")
    missing = {b for b, _ in qwen if not (C.PAGES / b).is_dir()}
    if missing:
        print(f"warning: {len(missing)} Qwen books have no lhh folder, skipped: {sorted(missing)[:5]}")
        qwen = {k: v for k, v in qwen.items() if k[0] not in missing}

    tiers, ocr_cache = assign_tiers(qwen, args.insurance)
    print(f"Qwen results: {len(qwen)} pages in {len({b for b, _ in qwen})} books")
    for t in (1, 2, 3, 4):
        n = sum(1 for v in tiers.values() if v == t)
        if n or t < 4:
            print(f"  tier {t}: {n:7} pages  (~${n * EST_TOKENS_PER_PAGE / 1e6 * PRICE_PER_M:.2f})")
    print(f"  to send: {len(tiers)} pages  (~${len(tiers) * EST_TOKENS_PER_PAGE / 1e6 * PRICE_PER_M:.2f})")

    out = ROOT / args.out
    jsonl = out / "results.jsonl"
    api = {k: r for k, r in C.load_results(jsonl).items() if r.get("p_character") is not None}
    todo = sorted(k for k in tiers if k not in api)
    print(f"  already scored: {len(api)}   remaining: {len(todo)}")
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)

    if todo:
        key = C.load_key()
        if not key:
            sys.exit("no API key: set OPENAI_API_KEY or put the key in openai.key")
        jobs, pages_cache, intro_cache = [], {}, {}
        for b, p in todo:
            if b not in pages_cache:
                pages_cache[b] = C.book_pages(b)
                intro_cache[b] = C.book_intro(ocr_cache[b], pages_cache[b])
            if p in pages_cache[b]:
                jobs.append((b, p, C.context_text(b, p, pages_cache[b], ocr_cache[b], intro_cache[b])))
        print(f"scoring {len(jobs)} pages with {args.jobs} workers")
        score_pages(jobs, jsonl, key, args.jobs, extra={k: {"tier": v} for k, v in tiers.items()})
        api = {k: r for k, r in C.load_results(jsonl).items() if r.get("p_character") is not None}

    write_merged(out, qwen, tiers, api)
    build_report(out, qwen=qwen)


if __name__ == "__main__":
    main()
