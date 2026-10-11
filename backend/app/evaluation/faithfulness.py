"""Answer faithfulness: is every claim in the answer supported by the evidence?

The answer is split into claims (sentences, citation markers stripped) and each claim
is judged against the retrieved chunks. Faithfulness = supported claims / claims.
A judge is any async callable ``(claim, context_chunks) -> bool``; the LLM-backed
judge lives in ``pipeline.py`` and ``make_lexical_judge`` here is a deterministic
offline fallback.
"""

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

ABSTAIN_PHRASE = "i don't have enough information"

Judge = Callable[[str, Sequence[str]], Awaitable[bool]]

_CITATION = re.compile(r"\s*\[\d+\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be been by can do does for from has have in is it its of on "
    "or that the their there this to was were which will with you your not no".split()
)


def is_abstention(answer: str) -> bool:
    return ABSTAIN_PHRASE in answer.lower().replace("’", "'")


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims with citation markers removed."""
    claims = []
    for sentence in _SENTENCE_SPLIT.split(answer):
        sentence = _CITATION.sub("", sentence).strip(" \t-*")
        if len(content_tokens(sentence)) >= 2:
            claims.append(sentence)
    return claims


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def support_score(claim: str, context: str) -> float:
    """Share of the claim's content words that appear in ``context``."""
    tokens = content_tokens(claim)
    if not tokens:
        return 1.0
    return len(tokens & content_tokens(context)) / len(tokens)


def make_lexical_judge(threshold: float = 0.7) -> Judge:
    """
    Deterministic judge: a claim is supported if at least ``threshold`` of its content
    words occur in a single chunk. It cannot detect contradictions or reasoning
    errors; use the LLM judge for that.
    """

    async def judge(claim: str, chunks: Sequence[str]) -> bool:
        return any(support_score(claim, chunk) >= threshold for chunk in chunks)

    return judge


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    # 1-based rank of the retrieved chunk that best overlaps the claim (None if no
    # chunk shares any content word); lets a reader trace the claim to its source.
    best_chunk_rank: int | None


@dataclass
class FaithfulnessResult:
    score: float  # supported / total claims; 1.0 when there are no claims
    claims: list[ClaimVerdict]

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if not c.supported]


async def evaluate_faithfulness(
    answer: str, chunks: Sequence[str], judge: Judge
) -> FaithfulnessResult:
    verdicts = []
    for claim in split_claims(answer):
        scores = [support_score(claim, c) for c in chunks]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        best_rank = best + 1 if best is not None and scores[best] > 0 else None
        verdicts.append(ClaimVerdict(claim, await judge(claim, chunks), best_rank))
    if not verdicts:
        return FaithfulnessResult(1.0, [])
    return FaithfulnessResult(
        sum(v.supported for v in verdicts) / len(verdicts), verdicts
    )
