# lianhuanhua-mao

Finding every page on which Mao Zedong appears in 1,598 scanned lianhuanhua
(连环画, Chinese picture books), 175,054 pages in all, in either of two forms:

1. **drawn as a character** in the panel, at any age: child, student, adult;
2. **as an image inside the scene**: a portrait on a wall, a carried portrait, a
   poster, a statue or bust, a chest badge, a banner, a book cover, a face in a sun.

The repository is a worked example of a pattern that applies to any "find every
occurrence of X in a large scanned collection" problem: a local vision-language
model sweeps everything, a second system is spent only on the pages where the first
is weak, and a human reviews the disagreements in a page that needs nothing but a
browser. The pipeline, the commands, the costs, and the things that went wrong are
all written down here. The images themselves are not included.

Outcome on this corpus: 1,607 flagged pages (0.9 %) in 525 books, 1,677 pages sent
to human review, everything else settled by the two systems agreeing. Roughly 20
GPU hours on one node and $1.70 of API calls.

## Why it is not trivial

- In the biographies Mao is a child or teenager, so his face alone does not
  identify him; the caption or the book context does.
- Captions name him on some pages and say only 他 on others. Many captions name
  him on pages where he is not drawn (one book names him on 62 pages and draws him
  on 2), and quotations like 毛主席说 are everywhere.
- In unrelated books he may appear only as a small framed portrait on a wall
  behind the main scene, a few dozen pixels wide and never mentioned in the text.

## Data layout

| Folder / file | Content |
|---|---|
| `lhh/<book>/00001.jpg …` | the page images, one folder per book, pages numbered from 1, about 1000 px wide. One drawn panel with its typeset caption |
| `ocr/<book>.jsonl` | stage 1 output, one line per page: `book`, `page`, `text` (`<无文字>` means no text) |
| `qwen/<book>.jsonl` | stage 2 output, one line per page: `mao_person`, `mao_image` (yes / no / unsure), `confidence`, `reason`, `box`, `book`, `page` |
| `supp/` | stage 3 output: `results.jsonl`, `results.csv`, `merged.csv`, `report.html`, `review.html`, and the reviewer's `review.csv` |
| `mao_pages.csv` | the final table: book, page, character, image, source |
| `openai.key` | API key for stage 3 (or set `OPENAI_API_KEY`); ignored by git |

Only the scripts and this file are versioned. The data folders are in `.gitignore`.

## Pipeline

```
lhh/          ──scripts/ocr.py        (GPU, Slurm)──►  ocr/
lhh/ + ocr/   ──scripts/detect.py     (GPU, Slurm)──►  qwen/
qwen/ + ocr/  ──scripts/supplement.py (OpenAI API)──►  supp/
supp/         ──scripts/review.py ──►  review.html / review_package.zip   human review
review.csv    ──scripts/review.py --finalize ──►  mao_pages.csv
```

`scripts/status.py` prints the coverage of every stage at any time.

| Script | Stage | Runs on | What it does |
|---|---|---|---|
| `scripts/ocr.py`, `run_ocr.sbatch` | 1 | GPU node | caption OCR with Qwen3-VL-8B through vLLM |
| `scripts/detect.py`, `run_detect.sbatch` | 2 | GPU node | per-page verdicts with Qwen3-VL-32B |
| `scripts/supplement.py` | 3 | anywhere with the API key | scores the hard pages with the OpenAI Decisions API |
| `scripts/review.py` | 4 | login node or laptop | review page, collaborator package, final table |
| `scripts/status.py` | — | anywhere | coverage and counts per stage |
| `scripts/report.py` | — | anywhere | card grid with the API scores for a results folder |
| `scripts/api_score.py` | optional | anywhere with the API key | full API scoring of given books or the whole corpus; holds the question texts |
| `scripts/common.py` | — | — | shared paths and loaders |

### Environment

The model stages need a GPU with about 70 GB of memory for the 32B model (we used
NVIDIA GH200 nodes, which are aarch64, so everything is pip-installable torch and
vLLM; nothing is compiled). `env.sh` keeps all caches next to the repository and
sets two vLLM variables for nodes without nvcc. The Slurm scripts source it.

```
curl -LsSf https://astral.sh/uv/install.sh | sh       # once, on a GPU node
source env.sh && uv sync                              # creates .venv with vllm
uv run hf download Qwen/Qwen3-VL-8B-Instruct          # 17 GB
uv run hf download Qwen/Qwen3-VL-32B-Instruct         # 67 GB
```

Stages 3 and 4 and the helper scripts are plain Python 3.9+. Only the API calls need
`httpx` (`pip install httpx`, or `uv run scripts/supplement.py`, which reads the
inline dependency block). `--dry-run`, `review.py`, `report.py` and `status.py` run
with a bare system Python.

The Slurm scripts take the account on the command line and assume a partition
called `gpu`; edit the `#SBATCH` lines for your cluster.

### 1. OCR

```
sbatch -A <account> scripts/run_ocr.sbatch
python3 scripts/status.py          # when the queue is empty: all pages present?
```

16 array tasks of one GPU each, about 9 pages/s per task, 4.5 GPU hours for the
corpus. Qwen3-VL-8B reads typeset captions accurately and returns each caption as
one paragraph, which the later stages need. Resumable: resubmitting fills missing
pages only.

### 2. VLM sweep

```
sbatch -A <account> scripts/run_detect.sbatch
python3 scripts/status.py
```

Qwen3-VL-32B looks at every page with its caption, the two previous captions and the
book title, and answers yes/no/unsure per form with a one-sentence reason, as JSON
enforced by a schema. About 3.3 pages/s per task, 15 GPU hours for the corpus.

What it is good at: recall on small portraits. It found 30-pixel portraits on stage
backdrops and wall portraits that the API missed. Its errors: it sometimes calls
Mao's father Mao in the biographies (reasoning from the caption), occasionally
counts a quotation board or a star on a flag as an image, and rejects figures whose
face is not recognisable even when the caption identifies them. Its `confidence` is
95 or 100 almost everywhere and carries no information.

Why 32B and not 8B: the 8B pilot treated "the caption names Mao" as "Mao is in the
panel" and boxed his father and his teacher. The 32B rejects the father's relatives,
his friends and other historical figures on the same pages and says unsure on
crowds. Why a plain prompt: rules written to argue the 8B out of its shortcuts did
not help; asking the natural question of a model that has the judgment did.

To rerun a few books, for a prompt change or a pilot:

```
uv run python scripts/detect.py --out qwen_pilot --books BOOK1 BOOK2
```

### 3. API supplement

The OpenAI Decisions API (`POST /v1/decisions`) answers three predicate questions per
page with calibrated probabilities: `mao_character`, `mao_image`, and a catch-all
`mao_any`. Each page is sent as the 1000 px image plus text from the OCR: the book's
title line (and synopsis for books that name Mao anywhere), the two previous
captions, the page's own caption. About 2,500 tokens per page, about 37 pages/s
with 16 workers.

The API is not run on everything. It is sent only where the sweep is weak:

| Tier | Pages | Why |
|---|---|---|
| 1 | Qwen said yes or unsure on either column | catch Qwen's false positives |
| 2 | Qwen said no, but the caption names Mao | catch missed drawn Mao |
| 3 | Qwen said no, but a clause of its reason affirms a portrait-like object | it saw something and talked itself out of it |
| 4 (`--insurance`, not used) | the rest of the books that name Mao anywhere | insurance against portraits Qwen never flagged |

Run: 6,923 pages, $1.70, no errors. Outcome against Qwen:

| Verdict | Pages |
|---|---|
| both yes | 866 |
| Qwen yes, API no | 741 |
| API yes, Qwen no | 39 |
| both no | 5,277 |

Of the 741 "Qwen yes, API no" pages, a random sample showed about half are real small
portraits the API scored at 0.00. So the API cannot be used to discard Qwen's flags;
those pages go to human review.

```
python3 scripts/supplement.py --dry-run     # counts per tier, no calls
uv run scripts/supplement.py                # tiers 1–3 into supp/ (resumable)
```

### 4. Human review

```
python3 scripts/review.py                   # writes supp/review.html
```

The page shows two groups, images linked from `lhh/`, nothing embedded:

- **uncertain** (950 here): the two systems disagree, an API score lies in the
  0.2–0.8 band, or Qwen said unsure. Each needs a decision.
- **agreed** (727 here): both say the person is present and no score is in the
  uncertain band. Pre-filled; browse and correct.

Everything else is settled without a human: the pages both systems rejected and the
168,000 pages neither flagged.

Keys: `1` drawn, `2` image, `3` both, `0` neither, `u` clear, `j`/`k` or arrows move,
`Enter` opens the page. Decisions live in the browser; **Export CSV** writes
`review.csv`; **Import CSV** restores it.

For a collaborator without the corpus:

```
python3 scripts/review.py --package         # review_package.zip: page + its images + README
```

They unzip, double-click `review.html`, review, export, and send back `review.csv`.
Two people can split by book with the book selector; their CSVs are concatenated.

### 5. Final table

```
cp <their file> supp/review.csv
python3 scripts/review.py --finalize        # -> mao_pages.csv
```

Rows come from the reviewer's decisions where they exist, otherwise from the API
scores (character = `p_character` ≥ 0.5; image = `p_image` ≥ 0.5, or `p_any` ≥ 0.5
when not drawn). Pages neither system flagged are absent.

## Adapting this to another person or object

Four places hold everything specific to Mao:

- `scripts/common.py`: `MAO_RE`, the names and titles the captions use.
- `scripts/detect.py`: `PROMPT`, the question the sweep model answers. Keep the
  structure, change the subject and the examples of what counts and what does not.
- `scripts/api_score.py`: `QUESTIONS`, the three predicate questions for the API.
- `scripts/review.py`: the `README` text that goes into the collaborator package.

The JSON field names (`mao_person`, `mao_image`, `p_character`, ...) are used across
the scripts; renaming them is optional and touches `common.py`, `supplement.py`,
`report.py` and `review.py`.

Expect the same failure modes: a mid-sized VLM answering from the text instead of
the picture, "text about X" being mistaken for "an image of X", and both systems
being noisy on exactly the small, hard cases. Pilot on three books before the full
run: one where the subject is the protagonist, one where it is only mentioned, and
one where it should only appear as a portrait.

## Cost and time on this corpus

| Stage | Pages | Compute | Wall time |
|---|---|---|---|
| OCR, Qwen3-VL-8B | 175,054 | 4.5 GPU h | 70 min on 16 GPUs |
| sweep, Qwen3-VL-32B | 175,054 | 15 GPU h | 1 h on 16 GPUs |
| API supplement | 6,923 | $1.70 | 3 min |
| human review | 1,677 | one afternoon | |

## Decisions and why

- **Images stay at 1000 px.** At 800 px and below the API loses a 30-pixel stage
  portrait. Image tokens are only ~770 of the ~2,500 per page; the rest is questions
  and context, so shrinking images saves little and costs recall.
- **Context text is deliberately short.** Trimming it saved only ~14 %; most of the
  per-request cost is fixed.
- **Three questions, not two.** The catch-all rescues pages where the model
  hesitates between the two forms (e.g. a face on a book cover).
- **Local model as the sweep, API as the check.** A full API pass would cost ~$44;
  the local model's recall on small portraits turned out to be as good or better, so
  the API is spent only on disagreement-prone pages.
- **Both systems are noisy on the same page.** The API's score for one wall portrait
  was 0.09, 0.00 and 0.68 across three identical runs. Hence the 0.2–0.8 band is
  reviewed rather than thresholded.
- **Caption mentions do not count.** Both systems were instructed accordingly.
- **Text-only objects are not images.** A quotation board, a slogan banner or a book
  spine with the name shows no face. The prompt says so; one or two still slip
  through and land in review.

## Known limitations

- Figures too small or too indistinct to recognise count only when the caption or
  book context identifies them; the two systems disagree on these. The reviewer
  decides.
- A portrait in a book whose captions never mention the subject is found only if the
  sweep flagged it; unflagged ones are not recoverable without a full API pass.
- Double-page spreads were scanned both as a spread and as two halves, so one panel
  can appear on two or three pages; each page is judged on its own.

## License

MIT. The scanned books are not part of this repository.
