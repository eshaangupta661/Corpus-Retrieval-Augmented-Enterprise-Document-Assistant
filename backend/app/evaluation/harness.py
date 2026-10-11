"""Run a labelled question set through a pipeline and score it."""

from collections import Counter
from dataclasses import asdict, dataclass, field

from app.evaluation import metrics
from app.evaluation.dataset import EvalExample
from app.evaluation.diagnosis import Diagnosis, Outcome, diagnose, relevance_gains
from app.evaluation.faithfulness import (
    FaithfulnessResult,
    Judge,
    evaluate_faithfulness,
    is_abstention,
)
from app.evaluation.types import EvalPipeline, PipelineOutput


@dataclass
class ExampleResult:
    example: EvalExample
    output: PipelineOutput
    diagnosis: Diagnosis
    faithfulness: FaithfulnessResult
    ndcg: float | None  # None for unanswerable questions (no labels to rank against)
    mrr: float | None
    recall: float | None
    hit: float | None


@dataclass
class Summary:
    k: int
    num_examples: int
    num_answerable: int
    ndcg: float
    mrr: float
    recall: float
    hit_rate: float
    faithfulness: float  # mean over non-abstaining answers
    outcomes: dict[str, int] = field(default_factory=dict)
    retrieval_failures: int = 0
    generation_failures: int = 0
    abstention_accuracy: float | None = None  # unanswerable questions only

    def to_dict(self) -> dict:
        return asdict(self)


async def evaluate_example(
    example: EvalExample, pipeline: EvalPipeline, judge: Judge, k: int
) -> ExampleResult:
    output = await pipeline.run(example.question)
    chunks = [item.content for item in output.retrieved]

    if is_abstention(output.answer):
        faithfulness = FaithfulnessResult(1.0, [])  # nothing asserted, nothing to check
    else:
        faithfulness = await evaluate_faithfulness(output.answer, chunks, judge)

    ndcg = mrr = recall = hit = None
    if example.answerable:
        gains = relevance_gains(example, output.retrieved)
        ndcg = metrics.ndcg_at_k(gains, k, [r.grade for r in example.relevant])
        mrr = metrics.reciprocal_rank(gains)
        recall = metrics.recall_at_k(gains, k, len(example.relevant))
        hit = metrics.hit_at_k(gains, k)

    diagnosis = diagnose(example, output, faithfulness.score, faithfulness.unsupported)
    return ExampleResult(
        example, output, diagnosis, faithfulness, ndcg, mrr, recall, hit
    )


async def run_evaluation(
    examples: list[EvalExample], pipeline: EvalPipeline, judge: Judge, k: int = 5
) -> tuple[list[ExampleResult], Summary]:
    results = [await evaluate_example(e, pipeline, judge, k) for e in examples]
    return results, summarize(results, k)


def summarize(results: list[ExampleResult], k: int) -> Summary:
    answerable = [r for r in results if r.example.answerable]
    unanswerable = [r for r in results if not r.example.answerable]
    answered = [r for r in results if not is_abstention(r.output.answer)]
    outcomes = Counter(r.diagnosis.outcome for r in results)

    return Summary(
        k=k,
        num_examples=len(results),
        num_answerable=len(answerable),
        ndcg=metrics.mean([r.ndcg for r in answerable if r.ndcg is not None]),
        mrr=metrics.mean([r.mrr for r in answerable if r.mrr is not None]),
        recall=metrics.mean([r.recall for r in answerable if r.recall is not None]),
        hit_rate=metrics.mean([r.hit for r in answerable if r.hit is not None]),
        faithfulness=metrics.mean([r.faithfulness.score for r in answered]),
        outcomes={o.value: n for o, n in outcomes.items()},
        retrieval_failures=sum(
            n for o, n in outcomes.items() if o.stage == "retrieval"
        ),
        generation_failures=sum(
            n for o, n in outcomes.items() if o.stage == "generation"
        ),
        abstention_accuracy=(
            sum(r.diagnosis.outcome == Outcome.CORRECT_ABSTENTION for r in unanswerable)
            / len(unanswerable)
            if unanswerable
            else None
        ),
    )
