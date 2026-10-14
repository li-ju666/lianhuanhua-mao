"""Stage 2: ask Qwen3-VL-32B, per page, whether Mao Zedong appears (a) drawn as a
character or (b) as an image inside the scene.  The page image is sent with its OCR
caption, the two previous captions and the book title.  GPU node, uv env.

    uv run python scripts/detect.py --shard 0 --nshards 1 [--books BOOK ...]

Writes qwen/<book>.jsonl, one line per page: mao_person, mao_image (yes/no/unsure),
confidence, reason, box, book, page.  Resumable like stage 1.  Normally launched
through scripts/run_detect.sbatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "mao_person": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "mao_image": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string", "maxLength": 200},
        "box": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 0, "maxItems": 4},
    },
    "required": ["mao_person", "mao_image", "confidence", "reason", "box"],
    "additionalProperties": False,
}

PROMPT = """这是中国连环画《{book}》（书名为拼音）中的一页。{mention}

前两页的说明文字：
{prev}

本页的说明文字：{caption}

请像一位熟悉中国现代史的读者那样，仔细看图并结合上下文，回答两个问题：
1. mao_person：画面里是否画了毛泽东本人（任何年龄，包括少年和青年时期）？
2. mao_image：画面场景里是否出现毛泽东的图像——指画出了他的面容或身形的东西：墙上的挂像、画像、海报、塑像、像章、横幅或旗帜上的头像等（有时很小）。只有文字的东西不算，例如语录牌、标语、横幅上的口号、书名《毛泽东选集》等，即使与毛泽东有关也回答 no。

每个问题回答 yes、no 或 unsure，给出 0-100 的把握程度和一句话理由。box 为毛泽东本人或图像在图中的位置 [x1,y1,x2,y2]，坐标范围 0-1000，没有则为 []。只输出 JSON。"""


def done_pages(jsonl: Path) -> set:
    if not jsonl.exists():
        return set()
    return {json.loads(line)["page"] for line in open(jsonl, encoding="utf-8")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.PAGES))
    ap.add_argument("--out", default=str(C.QWEN))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--books", nargs="*", default=None, help="only these book folders")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from PIL import Image

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    books = sorted(p for p in root.iterdir() if p.is_dir())[args.shard::args.nshards]
    if args.books:
        books = [root / b for b in args.books]
    print(f"shard {args.shard}/{args.nshards}: {len(books)} books", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem, limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=200,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))

    t0, n = time.time(), 0
    for bi, book in enumerate(books):
        jsonl = out / f"{book.name}.jsonl"
        have = done_pages(jsonl)
        names = sorted(p.name for p in book.iterdir() if p.suffix.lower() == ".jpg")
        todo = [p for p in names if p not in have]
        if not todo:
            continue
        caps = C.load_ocr(book.name)
        n_mention = sum(1 for t in caps.values() if C.names_mao(t))
        mention = f"全书 {len(names)} 页的说明文字中有 {n_mention} 页提到毛泽东。"

        def prompt_for(page):
            i = names.index(page)
            prev = [f"前第{j}页：{caps.get(names[i-j], '')}" for j in (2, 1) if i - j >= 0]
            return PROMPT.format(book=book.name, mention=mention, prev="\n".join(prev) or "（无）",
                                 caption=caps.get(page, "") or C.NO_TEXT)

        with open(jsonl, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                msgs = [[{"role": "user", "content": [
                    {"type": "image_pil", "image_pil": Image.open(book / p).convert("RGB")},
                    {"type": "text", "text": prompt_for(p)}]}] for p in chunk]
                outs = llm.chat(msgs, sp, use_tqdm=False)
                for p, o in zip(chunk, outs):
                    txt = o.outputs[0].text.strip()
                    try:
                        d = json.loads(txt)
                    except Exception:
                        d = {"mao_person": "unsure", "mao_image": "unsure", "confidence": 0,
                             "reason": "unparseable: " + txt[:100], "box": []}
                    d.update(book=book.name, page=p)
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
                n += len(chunk)
        el = time.time() - t0
        print(f"[{bi+1}/{len(books)}] {book.name}: {len(todo)} pages | total {n}, {n/el:.2f} p/s", flush=True)


if __name__ == "__main__":
    main()
