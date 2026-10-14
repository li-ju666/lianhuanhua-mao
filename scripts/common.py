"""Shared helpers for the pipeline scripts: repository paths, OCR loading, caption
context for the models, result loading.  Imported by the other scripts in this folder.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repository root
PAGES = ROOT / "lhh"     # <book>/00001.jpg ...            the page images
OCR = ROOT / "ocr"       # <book>.jsonl                    stage 1, scripts/ocr.py
QWEN = ROOT / "qwen"     # <book>.jsonl                    stage 2, scripts/detect.py
SUPP = ROOT / "supp"     # results.jsonl, merged.csv, ...  stage 3, scripts/supplement.py

MAO_RE = re.compile("毛泽东|毛主席|润之|毛委员|毛澤東|石三伢子")
NO_TEXT = "<无文字>"                  # what the OCR writes for a page without caption
THRESH = 0.5                          # API probability counted as "yes"
SCORE_FIELDS = ("p_character", "p_image", "p_any")


# ----------------------------------------------------------------- pages and OCR

def books() -> list[str]:
    return sorted(p.name for p in PAGES.iterdir() if p.is_dir())


def book_pages(book: str) -> list[str]:
    return sorted(p.name for p in (PAGES / book).iterdir() if p.suffix.lower() == ".jpg")


def load_ocr(book: str) -> dict[str, str]:
    """page filename -> caption text ('' for pages the OCR marked as empty)."""
    f = OCR / f"{book}.jsonl"
    if not f.exists():  # folder names can differ in Unicode normalisation (ü)
        want = unicodedata.normalize("NFC", book)
        f = next((c for c in OCR.glob("*.jsonl") if unicodedata.normalize("NFC", c.stem) == want), f)
    out = {}
    if f.exists():
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            t = r["text"].strip()
            out[r["page"]] = "" if t == NO_TEXT else t
    return out


def names_mao(text: str) -> bool:
    return bool(MAO_RE.search(text or ""))


# ----------------------------------------------------------------- model outputs

def load_jsonl_dir(d: Path) -> dict:
    """(book, page) -> row, over every <book>.jsonl in a folder; last line per page wins."""
    rows = {}
    for f in sorted(Path(d).glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "book" in r and "page" in r:
                rows[(r["book"], r["page"])] = r
    return rows


def load_qwen(qdir: Path = QWEN) -> dict:
    return load_jsonl_dir(qdir)


def qwen_says_yes(q: dict) -> bool:
    """A Qwen row counts as a flag when either form is yes or unsure."""
    return q.get("mao_person") in ("yes", "unsure") or q.get("mao_image") in ("yes", "unsure")


def load_results(jsonl: Path) -> dict:
    """(book, page) -> API result row; the last line for a page wins."""
    rows = {}
    if jsonl.exists():
        for line in jsonl.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["book"], r["page"])] = r
    return rows


def scored(rows: dict) -> set:
    return {k for k, r in rows.items() if r.get("p_character") is not None}


# ----------------------------------------------------------------- API context

def book_intro(ocr: dict[str, str], pages: list[str]) -> str:
    """Title line from the opening pages; the synopsis too, but only for books whose
    captions name Mao somewhere, which is where 他 needs resolving."""
    parts = [ocr[p].replace("\n", " ") for p in pages[:6] if ocr.get(p)]
    limit = 350 if any(names_mao(t) for t in ocr.values()) else 80
    return " / ".join(parts)[:limit]


def context_text(book: str, page: str, pages: list[str], ocr: dict[str, str], intro: str) -> str:
    i = pages.index(page)
    prev = [ocr[p].replace("\n", " ")[:120] for p in pages[max(0, i - 2):i] if ocr.get(p)]
    cur = ocr.get(page, "").replace("\n", " ")[:250]
    return (
        "A page of a Chinese lianhuanhua: one drawn panel with its caption below. "
        "Judge only what is drawn.\n"
        f"Book: {book}. {intro}\n"
        f"Previous captions: {' | '.join(prev) if prev else '(none)'}\n"
        f"This caption: {cur or '(no text)'}"
    )


def load_key() -> str:
    """OPENAI_API_KEY from the environment, else the key in openai.key at the repo root."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    f = ROOT / "openai.key"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip().strip('"').strip("'")
            if not line or line.startswith("#"):
                continue
            if "=" in line and not line.startswith("sk-"):
                line = line.split("=", 1)[1].strip().strip('"').strip("'")
            if line:
                return line
    return ""
