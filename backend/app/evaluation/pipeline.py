"""Adapters connecting the harness to the real RAG stack (needs DB + LLM)."""

from collections.abc import Sequence
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.faithfulness import Judge
from app.evaluation.types import PipelineOutput, RetrievedItem
from app.repositories.vector_repository import VectorRepository
from app.services.llm import llm_service
from app.services.prompt import prompt_service
from app.services.rag import RAGService
from app.services.retrieval import RetrievalService


def _item(chunk_id: str, content: str, metadata: dict | None) -> RetrievedItem:
    metadata = metadata or {}
    return RetrievedItem(
        chunk_id=chunk_id,
        source=metadata.get("title") or metadata.get("filename"),
        page=metadata.get("page_label") or metadata.get("page"),
        content=content,
    )


class RAGPipeline:
    """Runs the production retrieval + generation path for each question."""

    def __init__(
        self,
        db: AsyncSession,
        top_k: int = 5,
        rerank: bool = True,
        pool_size: int = 50,
    ):
        self.retrieval = RetrievalService(VectorRepository(db))
        self.rag = RAGService(self.retrieval)
        self.top_k = top_k
        self.rerank = rerank
        self.pool_size = pool_size

    async def run(self, question: str) -> PipelineOutput:
        result = await self.rag.generate_answer(
            question, top_k=self.top_k, rerank=self.rerank
        )
        retrieved = [
            _item(s["id"], s["content"], s["metadata"]) for s in result["sources"]
        ]

        # Un-reranked, wider candidate list used only to tell a ranking miss from a
        # retrieval miss during diagnosis.
        pool = await self.retrieval.hybrid_search(
            question, top_k=self.pool_size, rerank=False
        )
        candidate_pool = [
            _item(str(c.id), c.content, c.chunk_metadata) for c, _ in pool
        ]

        return PipelineOutput(
            answer=result["answer"],
            retrieved=retrieved,
            cited_ranks=[c["id"] for c in result["verified_citations"]],
            candidate_pool=candidate_pool,
        )


def make_llm_judge() -> Judge:
    """
    LLM-as-judge for claim support, reusing the production hallucination-check prompt.
    Unlike the runtime guard, errors propagate: a failed judge call must not silently
    count as "faithful".
    """

    async def judge(claim: str, chunks: Sequence[str]) -> bool:
        prompt = prompt_service.format_prompt(
            "hallucination_check",
            {"chunks": [SimpleNamespace(content=c) for c in chunks], "answer": claim},
        )
        verdict = await llm_service.chat_completion(
            prompt, temperature=0.0, max_tokens=10
        )
        return "hallucination" not in verdict.strip().lower()

    return judge
