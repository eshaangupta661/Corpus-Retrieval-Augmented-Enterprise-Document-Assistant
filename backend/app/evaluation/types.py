"""Data shapes exchanged between a pipeline under test and the harness."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RetrievedItem:
    chunk_id: str
    source: str | None
    page: int | str | None
    content: str


@dataclass
class PipelineOutput:
    answer: str
    retrieved: list[RetrievedItem]  # the final top-k, in rank order
    # 1-based ranks into ``retrieved`` that the answer cites ([1], [2], ...)
    cited_ranks: list[int] = field(default_factory=list)
    # Wider, un-reranked candidate list; lets the diagnosis tell "never retrieved"
    # apart from "retrieved but ranked out of the top-k".
    candidate_pool: list[RetrievedItem] = field(default_factory=list)


class EvalPipeline(Protocol):
    async def run(self, question: str) -> PipelineOutput: ...
