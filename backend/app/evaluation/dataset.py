"""Labelled evaluation questions.

One JSON object per line::

    {"id": "q1", "question": "How many vacation days do new hires get?",
     "relevant": [{"source": "hr_handbook.pdf", "page": 12}],
     "reference_answer": "New hires get 15 vacation days per year."}

``relevant`` lists the source pages that contain the answer; ``source`` is the
document title (the original upload filename). A label without
``page`` matches any page of that source. A question with an empty ``relevant``
list is *unanswerable*: the correct behaviour is to abstain.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RelevantSource:
    source: str
    page: int | None = None
    grade: float = 1.0

    def matches(self, source: str | None, page: int | str | None) -> bool:
        if source is None or source.strip().lower() != self.source.strip().lower():
            return False
        return self.page is None or str(page) == str(self.page)


@dataclass
class EvalExample:
    id: str
    question: str
    relevant: list[RelevantSource] = field(default_factory=list)
    reference_answer: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_dataset(path: str | Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
                examples.append(
                    EvalExample(
                        id=str(raw.get("id", f"q{len(examples) + 1}")),
                        question=raw["question"],
                        relevant=[
                            RelevantSource(
                                source=r["source"],
                                page=r.get("page"),
                                grade=float(r.get("grade", 1.0)),
                            )
                            for r in raw.get("relevant", [])
                        ],
                        reference_answer=raw.get("reference_answer"),
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"{path}:{line_no}: invalid example ({e})") from e
    return examples
