"""CLI: python -m app.evaluation --dataset eval/questions.jsonl [--k 5]"""

import argparse
import asyncio

from app.evaluation.dataset import load_dataset
from app.evaluation.faithfulness import make_lexical_judge
from app.evaluation.harness import run_evaluation
from app.evaluation.report import write_reports


async def _main(args: argparse.Namespace) -> None:
    # Imported here so `--help` and the pure-logic tests do not need DB/LLM config.
    from app.db.async_session import AsyncSessionLocal
    from app.evaluation.pipeline import RAGPipeline, make_llm_judge

    examples = load_dataset(args.dataset)
    judge = make_lexical_judge() if args.judge == "lexical" else make_llm_judge()

    async with AsyncSessionLocal() as db:
        pipeline = RAGPipeline(db, top_k=args.k, rerank=not args.no_rerank)
        results, summary = await run_evaluation(examples, pipeline, judge, k=args.k)

    json_path, md_path = write_reports(results, summary, args.out)
    print(
        f"nDCG@{args.k}={summary.ndcg:.3f}  MRR={summary.mrr:.3f}  "
        f"Recall@{args.k}={summary.recall:.3f}  faithfulness={summary.faithfulness:.3f}"
    )
    print(
        f"failures: retrieval={summary.retrieval_failures} "
        f"generation={summary.generation_failures}"
    )
    print(f"reports: {json_path}  {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation")
    parser.add_argument("--dataset", required=True, help="JSONL of labelled questions")
    parser.add_argument("--k", type=int, default=5, help="cutoff for @k metrics")
    parser.add_argument("--out", default="eval_reports", help="output directory")
    parser.add_argument(
        "--judge",
        choices=["llm", "lexical"],
        default="llm",
        help="faithfulness judge (lexical = offline, deterministic, coarser)",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="disable reranking (ablation)"
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
