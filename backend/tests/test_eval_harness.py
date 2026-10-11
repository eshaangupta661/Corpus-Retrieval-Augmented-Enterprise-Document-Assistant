import json

import pytest
from app.evaluation import metrics
from app.evaluation.dataset import EvalExample, RelevantSource, load_dataset
from app.evaluation.diagnosis import Outcome, relevance_gains
from app.evaluation.faithfulness import (
    evaluate_faithfulness,
    is_abstention,
    make_lexical_judge,
    split_claims,
)
from app.evaluation.harness import run_evaluation
from app.evaluation.report import to_markdown, write_reports
from app.evaluation.types import PipelineOutput, RetrievedItem

# ---------------------------------------------------------------- metrics


def test_reciprocal_rank():
    assert metrics.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([1, 0]) == 1.0
    assert metrics.reciprocal_rank([0, 0]) == 0.0
    assert metrics.reciprocal_rank([]) == 0.0


def test_ndcg_perfect_and_worse_orderings():
    assert metrics.ndcg_at_k([1, 0, 0], 3, [1]) == pytest.approx(1.0)
    # relevant item at rank 3: DCG = 1/log2(4) = 0.5, ideal = 1
    assert metrics.ndcg_at_k([0, 0, 1], 3, [1]) == pytest.approx(0.5)
    assert metrics.ndcg_at_k([0, 0, 0], 3, [1]) == 0.0


def test_ndcg_uses_labels_for_ideal_not_retrieved():
    # Two relevant labels but only one retrieved at rank 1: ideal DCG covers both.
    expected = 1.0 / (1.0 + 1 / 1.5849625007211562)
    assert metrics.ndcg_at_k([1, 0], 2, [1, 1]) == pytest.approx(expected)


def test_ndcg_graded_relevance():
    # grade 2 outranking grade 1 beats the reverse
    good = metrics.ndcg_at_k([2, 1], 2, [2, 1])
    bad = metrics.ndcg_at_k([1, 2], 2, [2, 1])
    assert good == pytest.approx(1.0)
    assert bad < good


def test_ndcg_respects_cutoff_and_empty_labels():
    assert metrics.ndcg_at_k([0, 0, 1], 2, [1]) == 0.0
    assert metrics.ndcg_at_k([1], 3, []) == 0.0


def test_recall_and_hit():
    assert metrics.recall_at_k([1, 0, 1], 3, 4) == 0.5
    assert metrics.recall_at_k([1], 3, 0) == 0.0
    assert metrics.hit_at_k([0, 0, 1], 2) == 0.0
    assert metrics.hit_at_k([0, 0, 1], 3) == 1.0


# ---------------------------------------------------------------- labels


def item(i, source="handbook.pdf", page=1, content="text"):
    return RetrievedItem(f"c{i}", source, page, content)


def test_relevant_source_matching():
    assert RelevantSource("Handbook.pdf", 3).matches("handbook.pdf", 3)
    assert RelevantSource("handbook.pdf", 3).matches("handbook.pdf", "3")
    assert not RelevantSource("handbook.pdf", 3).matches("handbook.pdf", 4)
    assert RelevantSource("handbook.pdf").matches("handbook.pdf", 99)
    assert not RelevantSource("handbook.pdf").matches(None, 1)


def test_each_label_is_credited_once():
    ex = EvalExample("q", "?", [RelevantSource("handbook.pdf", 1)])
    # three chunks from the same relevant page must not triple-count the gain
    assert relevance_gains(ex, [item(1), item(2), item(3)]) == [1.0, 0.0, 0.0]


def test_load_dataset(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        "# comment\n"
        + json.dumps(
            {
                "id": "a",
                "question": "Q?",
                "relevant": [{"source": "x.pdf", "page": 2}],
                "reference_answer": "R",
            }
        )
        + "\n"
        + json.dumps({"question": "Unanswerable?"})
        + "\n"
    )
    a, b = load_dataset(path)
    assert a.answerable and a.relevant[0].page == 2 and a.reference_answer == "R"
    assert not b.answerable and b.id == "q2"


def test_load_dataset_reports_bad_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"no_question": 1}\n')
    with pytest.raises(ValueError, match=":1:"):
        load_dataset(path)


# ---------------------------------------------------------------- faithfulness


def test_split_claims_strips_citations():
    claims = split_claims("New hires get 15 vacation days [1]. Unused days expire [2].")
    assert claims == ["New hires get 15 vacation days.", "Unused days expire."]


def test_is_abstention_handles_curly_apostrophe():
    assert is_abstention("I don't have enough information to answer this question.")
    assert is_abstention("I don’t have enough information to answer.")
    assert not is_abstention("Employees get 15 days.")


async def test_faithfulness_flags_claim_from_model_memory():
    chunks = ["Employees accrue 15 vacation days per year."]
    answer = (
        "Employees accrue 15 vacation days per year [1]. "
        "Contractors receive unlimited sabbatical leave."
    )
    result = await evaluate_faithfulness(answer, chunks, make_lexical_judge())
    assert result.score == pytest.approx(0.5)
    assert [c.claim for c in result.unsupported] == [
        "Contractors receive unlimited sabbatical leave."
    ]


async def test_faithfulness_empty_answer_is_vacuously_faithful():
    result = await evaluate_faithfulness("", ["ctx"], make_lexical_judge())
    assert result.score == 1.0


# ---------------------------------------------------------------- end to end

DOC = "handbook.pdf"
VACATION = "Employees accrue 15 vacation days per year."
PARKING = "Visitor parking is located behind building two."
ABSTAIN = "I don't have enough information to answer this question."


class ScriptedPipeline:
    """Returns a canned PipelineOutput per question."""

    def __init__(self, outputs):
        self.outputs = outputs

    async def run(self, question):
        return self.outputs[question]


def vacation_example(qid="vac"):
    return EvalExample(
        qid,
        f"vacation {qid}",
        [RelevantSource(DOC, 1)],
        reference_answer="Employees accrue 15 vacation days per year.",
    )


async def run_one(example, output):
    pipeline = ScriptedPipeline({example.question: output})
    results, summary = await run_evaluation(
        [example], pipeline, make_lexical_judge(), k=3
    )
    return results[0], summary


async def test_ok_case():
    out = PipelineOutput(
        answer=f"{VACATION[:-1]} [1].",
        retrieved=[item(1, DOC, 1, VACATION), item(2, DOC, 2, PARKING)],
        cited_ranks=[1],
    )
    result, summary = await run_one(vacation_example(), out)
    assert result.diagnosis.outcome == Outcome.OK
    assert result.ndcg == pytest.approx(1.0)
    assert summary.mrr == 1.0 and summary.faithfulness == 1.0
    assert summary.retrieval_failures == summary.generation_failures == 0


async def test_retrieval_miss_is_a_retrieval_failure():
    out = PipelineOutput(
        answer=ABSTAIN,
        retrieved=[item(1, DOC, 2, PARKING)],
        candidate_pool=[item(1, DOC, 2, PARKING)],
    )
    result, summary = await run_one(vacation_example(), out)
    assert result.diagnosis.outcome == Outcome.RETRIEVAL_MISS
    assert result.diagnosis.outcome.stage == "retrieval"
    assert summary.retrieval_failures == 1 and summary.generation_failures == 0
    assert result.ndcg == 0.0 and result.mrr == 0.0


async def test_ranking_miss_when_evidence_is_only_in_candidate_pool():
    out = PipelineOutput(
        answer=ABSTAIN,
        retrieved=[item(1, DOC, 2, PARKING)],
        candidate_pool=[item(1, DOC, 2, PARKING), item(2, DOC, 1, VACATION)],
    )
    result, _ = await run_one(vacation_example(), out)
    assert result.diagnosis.outcome == Outcome.RANKING_MISS
    assert result.diagnosis.outcome.stage == "retrieval"


async def test_abstaining_despite_evidence_is_a_generation_failure():
    out = PipelineOutput(answer=ABSTAIN, retrieved=[item(1, DOC, 1, VACATION)])
    result, summary = await run_one(vacation_example(), out)
    assert result.diagnosis.outcome == Outcome.ABSTAINED_DESPITE_EVIDENCE
    assert summary.generation_failures == 1 and summary.retrieval_failures == 0
    # retrieval itself was perfect
    assert result.ndcg == pytest.approx(1.0)


async def test_unfaithful_answer_is_a_generation_failure():
    out = PipelineOutput(
        answer="Contractors receive unlimited sabbatical leave every quarter [1].",
        retrieved=[item(1, DOC, 1, VACATION)],
        cited_ranks=[1],
    )
    result, _ = await run_one(vacation_example(), out)
    assert result.diagnosis.outcome == Outcome.UNFAITHFUL_ANSWER
    assert result.diagnosis.outcome.stage == "generation"
    assert result.diagnosis.unsupported_claims


async def test_wrong_citation_when_grounded_answer_cites_irrelevant_chunk():
    out = PipelineOutput(
        answer=f"{VACATION[:-1]} [2].",
        retrieved=[item(1, DOC, 1, VACATION), item(2, DOC, 2, PARKING)],
        cited_ranks=[2],
    )
    result, _ = await run_one(vacation_example(), out)
    assert result.diagnosis.outcome == Outcome.WRONG_CITATION


async def test_incomplete_answer_against_reference():
    ex = EvalExample(
        "inc",
        "vacation inc",
        [RelevantSource(DOC, 1)],
        reference_answer="Employees accrue vacation days annually; unused days carry "
        "over for twelve months before expiring.",
    )
    out = PipelineOutput(
        answer="Employees accrue vacation days [1].",
        retrieved=[
            item(
                1,
                DOC,
                1,
                "Employees accrue vacation days annually; unused days carry over for "
                "twelve months before expiring.",
            )
        ],
        cited_ranks=[1],
    )
    result, _ = await run_one(ex, out)
    assert result.diagnosis.outcome == Outcome.INCOMPLETE_ANSWER


async def test_unanswerable_question():
    ex = EvalExample("u1", "unanswerable one")
    abstain = PipelineOutput(answer=ABSTAIN, retrieved=[item(1, DOC, 2, PARKING)])
    bluff = PipelineOutput(
        answer="Visitor parking is located behind building two [1].",
        retrieved=[item(1, DOC, 2, PARKING)],
        cited_ranks=[1],
    )
    good, _ = await run_one(ex, abstain)
    bad, summary = await run_one(EvalExample("u2", "unanswerable two"), bluff)
    assert good.diagnosis.outcome == Outcome.CORRECT_ABSTENTION
    assert good.ndcg is None
    assert bad.diagnosis.outcome == Outcome.ANSWERED_UNANSWERABLE
    assert summary.abstention_accuracy == 0.0


async def test_summary_splits_failures_and_reports_render(tmp_path):
    examples = [vacation_example("a"), vacation_example("b"), vacation_example("c")]
    ok = PipelineOutput(
        answer=f"{VACATION[:-1]} [1].",
        retrieved=[item(1, DOC, 1, VACATION)],
        cited_ranks=[1],
    )
    miss = PipelineOutput(answer=ABSTAIN, retrieved=[item(1, DOC, 2, PARKING)])
    refused = PipelineOutput(answer=ABSTAIN, retrieved=[item(1, DOC, 1, VACATION)])
    pipeline = ScriptedPipeline(
        {
            examples[0].question: ok,
            examples[1].question: miss,
            examples[2].question: refused,
        }
    )
    results, summary = await run_evaluation(examples, pipeline, make_lexical_judge(), 3)

    assert summary.retrieval_failures == 1
    assert summary.generation_failures == 1
    assert summary.mrr == pytest.approx(2 / 3)
    assert summary.hit_rate == pytest.approx(2 / 3)

    md = to_markdown(results, summary)
    assert "retrieval_miss" in md and "abstained_despite_evidence" in md
    assert "handbook.pdf, p.2" in md

    json_path, md_path = write_reports(results, summary, tmp_path / "out")
    data = json.loads(json_path.read_text())
    assert data["summary"]["retrieval_failures"] == 1
    assert {r["failure_stage"] for r in data["results"]} == {
        None,
        "retrieval",
        "generation",
    }
    assert md_path.exists()
