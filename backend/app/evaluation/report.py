"""Render evaluation results as JSON (machine) and Markdown (human)."""

import json
from pathlib import Path

from app.evaluation.harness import ExampleResult, Summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _source_label(item) -> str:
    page = f", p.{item.page}" if item.page is not None else ""
    return f"{item.source or 'unknown'}{page}"


def result_to_dict(r: ExampleResult) -> dict:
    return {
        "id": r.example.id,
        "question": r.example.question,
        "answerable": r.example.answerable,
        "outcome": r.diagnosis.outcome.value,
        "failure_stage": r.diagnosis.outcome.stage,
        "explanation": r.diagnosis.explanation,
        "ndcg": r.ndcg,
        "mrr": r.mrr,
        "recall": r.recall,
        "faithfulness": r.faithfulness.score,
        "answer": r.output.answer,
        "cited_ranks": r.output.cited_ranks,
        "expected": [{"source": s.source, "page": s.page} for s in r.example.relevant],
        "retrieved": [
            {"rank": i, "chunk_id": c.chunk_id, "source": c.source, "page": c.page}
            for i, c in enumerate(r.output.retrieved, start=1)
        ],
        "unsupported_claims": [
            {"claim": c.claim, "closest_chunk_rank": c.best_chunk_rank}
            for c in r.faithfulness.unsupported
        ],
    }


def to_markdown(results: list[ExampleResult], summary: Summary) -> str:
    s = summary
    lines = [
        "# RAG evaluation report",
        "",
        f"{s.num_examples} questions ({s.num_answerable} answerable), k={s.k}",
        "",
        "## Retrieval (answerable questions)",
        "",
        f"| nDCG@{s.k} | MRR | Recall@{s.k} | Hit@{s.k} |",
        "|---|---|---|---|",
        f"| {_fmt(s.ndcg)} | {_fmt(s.mrr)} | {_fmt(s.recall)} | {_fmt(s.hit_rate)} |",
        "",
        "## Generation",
        "",
        f"- Mean faithfulness (non-abstaining answers): **{_fmt(s.faithfulness)}**",
        f"- Abstention accuracy on unanswerable questions: "
        f"**{_fmt(s.abstention_accuracy)}**",
        "",
        "## Failure attribution",
        "",
        f"- Retrieval failures: **{s.retrieval_failures}**",
        f"- Generation failures: **{s.generation_failures}**",
        "",
        "| Outcome | Count |",
        "|---|---|",
        *[f"| {o} | {n} |" for o, n in sorted(s.outcomes.items())],
    ]

    failures = [r for r in results if r.diagnosis.outcome.stage is not None]
    if failures:
        lines += ["", "## Failure traces", ""]
    for r in failures:
        d = r.diagnosis
        lines += [
            f"### {r.example.id} — {d.outcome.value} ({d.outcome.stage})",
            "",
            f"**Q:** {r.example.question}",
            "",
            f"**Why:** {d.explanation}",
            "",
            "**Expected:** "
            + (
                ", ".join(
                    f"{x.source}" + (f", p.{x.page}" if x.page is not None else "")
                    for x in r.example.relevant
                )
                or "(unanswerable)"
            ),
            "",
            "**Retrieved:** "
            + "; ".join(
                f"[{i}] {_source_label(c)}"
                for i, c in enumerate(r.output.retrieved, start=1)
            ),
            "",
            f"**Answer:** {r.output.answer}",
            "",
        ]
        for c in r.faithfulness.unsupported:
            closest = (
                f"closest chunk [{c.best_chunk_rank}]"
                if c.best_chunk_rank
                else "no overlapping chunk"
            )
            lines.append(f'- Unsupported: "{c.claim}" ({closest})')
        lines.append("")
    return "\n".join(lines)


def write_reports(
    results: list[ExampleResult], summary: Summary, out_dir: str | Path
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "eval_report.json"
    md_path = out / "eval_report.md"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary.to_dict(),
                "results": [result_to_dict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(to_markdown(results, summary), encoding="utf-8")
    return json_path, md_path
