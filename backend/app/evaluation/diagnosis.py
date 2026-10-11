"""Failure attribution: was a bad answer a retrieval failure or a generation failure?

The two need different fixes (chunking / embeddings / reranker vs. prompt / model), so
the split has to be made before recommending either. The decision is made in order:

1. Did retrieval surface the labelled evidence in the final top-k?  If not, it is a
   *retrieval* failure, which is further split into "never in the candidate pool"
   (indexing/embedding/keyword problem) and "in the pool but ranked out" (ranking).
2. If the evidence *was* retrieved, any wrong answer is a *generation* failure:
   abstaining despite the evidence, unsupported claims, citing the wrong sources, or
   omitting what the reference answer says.
"""

from dataclasses import dataclass, field
from enum import Enum

from app.evaluation.dataset import EvalExample
from app.evaluation.faithfulness import (
    ClaimVerdict,
    content_tokens,
    is_abstention,
)
from app.evaluation.types import PipelineOutput, RetrievedItem

FAITHFULNESS_THRESHOLD = 0.8
REFERENCE_RECALL_THRESHOLD = 0.5


class Outcome(str, Enum):
    OK = "ok"
    CORRECT_ABSTENTION = "correct_abstention"
    # retrieval failures
    RETRIEVAL_MISS = "retrieval_miss"
    RANKING_MISS = "ranking_miss"
    # generation failures (the evidence was in the context)
    ABSTAINED_DESPITE_EVIDENCE = "abstained_despite_evidence"
    UNFAITHFUL_ANSWER = "unfaithful_answer"
    WRONG_CITATION = "wrong_citation"
    INCOMPLETE_ANSWER = "incomplete_answer"
    ANSWERED_UNANSWERABLE = "answered_unanswerable"

    @property
    def stage(self) -> str | None:
        """'retrieval', 'generation', or None when the case passed."""
        if self in (Outcome.OK, Outcome.CORRECT_ABSTENTION):
            return None
        if self in (Outcome.RETRIEVAL_MISS, Outcome.RANKING_MISS):
            return "retrieval"
        return "generation"


@dataclass
class Diagnosis:
    outcome: Outcome
    explanation: str
    # Rank (1-based) of every retrieved chunk that matches a label.
    relevant_ranks: list[int] = field(default_factory=list)
    unsupported_claims: list[ClaimVerdict] = field(default_factory=list)


def relevance_gains(example: EvalExample, items: list[RetrievedItem]) -> list[float]:
    """
    Grade of each retrieved item. Every label can be satisfied once (by the first
    chunk that matches it), so several chunks from the same relevant page do not
    inflate the gain.
    """
    claimed: set[int] = set()
    gains = []
    for item in items:
        gain = 0.0
        for i, label in enumerate(example.relevant):
            if i not in claimed and label.matches(item.source, item.page):
                claimed.add(i)
                gain = label.grade
                break
        gains.append(gain)
    return gains


def _matches_any_label(example: EvalExample, item: RetrievedItem) -> bool:
    return any(label.matches(item.source, item.page) for label in example.relevant)


def reference_recall(answer: str, reference: str) -> float:
    ref_tokens = content_tokens(reference)
    if not ref_tokens:
        return 1.0
    return len(ref_tokens & content_tokens(answer)) / len(ref_tokens)


def diagnose(
    example: EvalExample,
    output: PipelineOutput,
    faithfulness_score: float,
    unsupported: list[ClaimVerdict],
) -> Diagnosis:
    abstained = is_abstention(output.answer)

    if not example.answerable:
        if abstained:
            return Diagnosis(
                Outcome.CORRECT_ABSTENTION, "Correctly declined to answer."
            )
        return Diagnosis(
            Outcome.ANSWERED_UNANSWERABLE,
            "No source contains the answer, but the model answered instead of "
            "abstaining.",
            unsupported_claims=unsupported,
        )

    relevant_ranks = [
        rank
        for rank, item in enumerate(output.retrieved, start=1)
        if _matches_any_label(example, item)
    ]

    # --- Stage 1: retrieval -------------------------------------------------------
    if not relevant_ranks:
        in_pool = any(_matches_any_label(example, i) for i in output.candidate_pool)
        if in_pool:
            return Diagnosis(
                Outcome.RANKING_MISS,
                "Labelled evidence was in the candidate pool but ranked below the "
                "top-k; investigate the reranker / fusion weights.",
            )
        return Diagnosis(
            Outcome.RETRIEVAL_MISS,
            "Labelled evidence was never retrieved; investigate chunking, "
            "embeddings and keyword matching.",
        )

    # --- Stage 2: generation (evidence was in the context) -------------------------
    common = {"relevant_ranks": relevant_ranks, "unsupported_claims": unsupported}

    if abstained:
        return Diagnosis(
            Outcome.ABSTAINED_DESPITE_EVIDENCE,
            f"Evidence was retrieved at rank(s) {relevant_ranks} but the model "
            "declined to answer.",
            **common,
        )
    if faithfulness_score < FAITHFULNESS_THRESHOLD:
        return Diagnosis(
            Outcome.UNFAITHFUL_ANSWER,
            f"Evidence was retrieved but {len(unsupported)} claim(s) are not "
            "supported by the retrieved context.",
            **common,
        )
    if not set(output.cited_ranks) & set(relevant_ranks):
        return Diagnosis(
            Outcome.WRONG_CITATION,
            f"Answer is grounded but cites rank(s) {sorted(set(output.cited_ranks))} "
            f"instead of the relevant rank(s) {relevant_ranks}.",
            **common,
        )
    if (
        example.reference_answer
        and reference_recall(output.answer, example.reference_answer)
        < REFERENCE_RECALL_THRESHOLD
    ):
        return Diagnosis(
            Outcome.INCOMPLETE_ANSWER,
            "Answer is grounded and cited but omits most of the reference answer.",
            **common,
        )
    return Diagnosis(Outcome.OK, "Retrieved, grounded and cited correctly.", **common)
