import pytest
from app.services.chunking import ChunkingService


@pytest.fixture
def chunking_service():
    return ChunkingService(chunk_size=100, chunk_overlap=20)


def test_split_text_basic(chunking_service):
    text = "A" * 250
    chunks = chunking_service.split_text(text)

    # With chunk_size=100 and overlap=20:
    # 1. 0-100
    # 2. 80-180
    # 3. 160-260 (but text ends at 250)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)


def test_create_chunks_with_metadata(chunking_service):
    text = "Some sample text to be chunked. " * 10
    metadata = {"doc_id": "test-123"}
    chunks_with_meta = chunking_service.create_chunks_with_metadata(text, metadata)

    assert len(chunks_with_meta) > 1
    for i, item in enumerate(chunks_with_meta):
        assert "content" in item
        assert "metadata" in item
        assert item["metadata"]["doc_id"] == "test-123"
        assert item["metadata"]["chunk_index"] == i


def test_chunk_overlap(chunking_service):
    # Create a text where we can see the overlap clearly
    # Each sentence is about 25 chars.
    text = "Sentence one is here now. "  # 26
    text += "Sentence two is also here. "  # 27
    text += "Sentence three is the next. "  # 28
    text += "Sentence four is over here. "  # 28

    # total length ~ 109

    custom_service = ChunkingService(chunk_size=50, chunk_overlap=10)
    chunks = custom_service.split_text(text)

    # Chunk 1 should end around char 50
    # Chunk 2 should start 10 chars before Chunk 1 ends
    assert len(chunks) >= 2
    # Check that some text from end of chunk 0 is in start of chunk 1
    last_10_of_c0 = chunks[0][-10:]
    first_few_of_c1 = chunks[1][:20]
    assert any(word in first_few_of_c1 for word in last_10_of_c0.split())


def test_create_chunks_from_pages_tags_page_and_never_spans_pages(chunking_service):
    pages = [(1, "alpha " * 40), (2, "beta " * 40), (None, "gamma")]
    chunks = chunking_service.create_chunks_from_pages(pages, {"title": "T"})

    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c["metadata"]["title"] == "T"
        page = c["metadata"].get("page")
        if page == 1:
            assert "beta" not in c["content"]
        if page == 2:
            assert "alpha" not in c["content"]
    assert {c["metadata"].get("page") for c in chunks} == {1, 2, None}
