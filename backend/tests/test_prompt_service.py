from types import SimpleNamespace

from app.services.prompt import prompt_service


def _chunk(metadata):
    return SimpleNamespace(content="Vacation is 15 days.", chunk_metadata=metadata)


def test_rag_prompt_labels_each_source_with_title_and_page():
    messages = prompt_service.format_prompt(
        "rag_answer",
        {
            "chunks": [_chunk({"title": "HR Handbook", "page": 12}), _chunk(None)],
            "question": "How many vacation days?",
            "history": [],
        },
    )
    user = messages[1]["content"]
    assert "Source Title: HR Handbook" in user
    assert "Source Page: 12" in user
    assert "Source Title: Unknown" in user
    assert user.count("Source Page:") == 1


def test_rag_prompt_forbids_outside_knowledge():
    system = prompt_service.format_prompt(
        "rag_answer", {"chunks": [], "question": "q", "history": []}
    )[0]["content"]
    assert "ONLY" in system and "training" in system
