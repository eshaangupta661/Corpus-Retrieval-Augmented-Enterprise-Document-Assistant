# RAG Evaluation

The harness in `backend/app/evaluation` answers three questions about the pipeline:

1. **Does retrieval surface the right source pages?** (nDCG, MRR, Recall, Hit)
2. **Is the answer grounded in what was retrieved?** (faithfulness)
3. **When it fails, which stage failed?** (retrieval vs. generation)

Question 3 matters because the fixes differ: retrieval failures call for changes to chunking, embeddings, fusion or the reranker; generation failures call for changes to the prompt, model or context size.

## Running it

```bash
cd backend
python -m app.evaluation --dataset eval/questions.jsonl --k 5 --out eval_reports
```

| Flag | Meaning |
|---|---|
| `--dataset` | JSONL file of labelled questions (below) |
| `--k` | cutoff for nDCG@k / Recall@k / Hit@k (default 5) |
| `--judge llm\|lexical` | faithfulness judge. `llm` (default) reuses the `hallucination_check` prompt per claim; `lexical` is offline and deterministic but cannot catch contradictions |
| `--no-rerank` | disable the reranker, to measure what it adds |
| `--out` | directory for `eval_report.json` and `eval_report.md` |

The documents must already be uploaded and processed. Labels refer to documents by **title** (the original filename) and **page**.

## Dataset format

One JSON object per line (`backend/eval/example_questions.jsonl` shows the shape; replace its labels with ones for your own corpus):

```json
{"id": "vac-1", "question": "How many vacation days do new hires get?",
 "relevant": [{"source": "hr_handbook.pdf", "page": 12}],
 "reference_answer": "New hires get 15 vacation days per year."}
```

- `relevant`: the page(s) that contain the answer. Omit `page` to accept any page of the document. An optional `grade` (default 1) allows graded relevance for nDCG.
- `reference_answer` (optional): used to flag answers that are grounded but incomplete.
- An empty or missing `relevant` marks the question **unanswerable**; the correct behaviour is to abstain.

## Metrics

Computed over answerable questions, on the final top-k chunks:

- **nDCG@k**: rank-weighted gain, normalised by the ideal ranking built from the labels (not from what was retrieved). Each label is credited once, so several chunks from one relevant page do not inflate the score.
- **MRR**: mean of 1 / rank of the first relevant chunk.
- **Recall@k / Hit@k**: share of labelled pages found / whether any was found.
- **Faithfulness**: the answer is split into sentence-level claims (citation markers removed); faithfulness is the fraction judged supported by the retrieved chunks. Abstentions are excluded from the mean.
- **Abstention accuracy**: share of unanswerable questions where the model declined.

## Failure attribution

Each question gets exactly one outcome, decided in this order:

| Stage | Outcome | Meaning |
|---|---|---|
| retrieval | `retrieval_miss` | labelled page was not in the candidate pool at all |
| retrieval | `ranking_miss` | it was in the candidate pool but ranked below the top-k |
| generation | `abstained_despite_evidence` | evidence was retrieved but the model declined to answer |
| generation | `unfaithful_answer` | evidence was retrieved but claims are unsupported by the context |
| generation | `wrong_citation` | answer is grounded but cites chunks other than the relevant ones |
| generation | `incomplete_answer` | grounded and cited, but misses most of the reference answer |
| generation | `answered_unanswerable` | no source has the answer, yet the model answered |
| pass | `ok`, `correct_abstention` | |

The Markdown report lists every failure with the expected pages, the retrieved sources by rank, the answer, and each unsupported claim with the retrieved chunk it overlaps most, so a failing answer can be traced back to its chunks.

## Limits

- The lexical judge is a coarse word-overlap check; use the LLM judge for real runs. The LLM judge is itself a model and can be wrong, so spot-check its verdicts.
- Results are only as good as the labels. Keep the question set small enough to label carefully, and include unanswerable questions.
- `candidate_pool` for the `ranking_miss` check is the un-reranked top 50 fused candidates.
