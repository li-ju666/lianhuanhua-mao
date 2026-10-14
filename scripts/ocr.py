"""Stage 1: OCR every page caption with Qwen3-VL served by vLLM.  GPU node, uv env.

    uv run python scripts/ocr.py --shard 0 --nshards 1 [--books BOOK ...]

Writes ocr/<book>.jsonl, one line per page: {"book", "page", "text"}.  Pages already
written are skipped, so the script is resumable and the Slurm array can be resubmitted.
Normally launched through scripts/run_ocr.sbatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PROMPT = (
    "这是一页连环画。请只输出页面上印刷的说明文字（图画旁边或下面的文字），"
    "一字不差地转写，包括开头的页码数字。不要描述图画，不要加任何解释。"
    f"如果页面上没有印刷文字，只输出 {C.NO_TEXT}。"
)


def done_pages(jsonl: Path) -> set:
    if not jsonl.exists():
        return set()
    return {json.loads(line)["page"] for line in open(jsonl, encoding="utf-8")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.PAGES))
    ap.add_argument("--out", default=str(C.OCR))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--books", nargs="*", default=None, help="only these book folders")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from PIL import Image

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    books = sorted(p for p in root.iterdir() if p.is_dir())[args.shard::args.nshards]
    if args.books:
        books = [root / b for b in args.books]
    print(f"shard {args.shard}/{args.nshards}: {len(books)} books", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem, limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=400)

    t0, n = time.time(), 0
    for bi, book in enumerate(books):
        jsonl = out / f"{book.name}.jsonl"
        have = done_pages(jsonl)
        todo = [p for p in sorted(book.iterdir()) if p.suffix.lower() == ".jpg" and p.name not in have]
        if not todo:
            continue
        with open(jsonl, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                msgs = [[{"role": "user", "content": [
                    {"type": "image_pil", "image_pil": Image.open(p).convert("RGB")},
                    {"type": "text", "text": PROMPT}]}] for p in chunk]
                outs = llm.chat(msgs, sp, use_tqdm=False)
                for p, o in zip(chunk, outs):
                    f.write(json.dumps({"book": book.name, "page": p.name,
                                        "text": o.outputs[0].text.strip()}, ensure_ascii=False) + "\n")
                f.flush()
                n += len(chunk)
        el = time.time() - t0
        print(f"[{bi+1}/{len(books)}] {book.name}: {len(todo)} pages | total {n}, {n/el:.2f} p/s", flush=True)


if __name__ == "__main__":
    main()
