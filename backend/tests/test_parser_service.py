from unittest.mock import MagicMock, patch

import pytest
from app.services.parser import parser_service


def test_parse_txt():
    content = b"Hello, world!"
    text = parser_service.parse(content, "txt")
    assert text == "Hello, world!"


def test_parse_unsupported():
    content = b"content"
    with pytest.raises(ValueError):
        parser_service.parse(content, "unknown")


def test_parse_pdf_structured(monkeypatch):
    # Mock elements returned by partition_pdf
    mock_title = MagicMock()
    mock_title.category = "Title"
    mock_title.text = "Doc Title"

    mock_table = MagicMock()
    mock_table.category = "Table"
    mock_table.text = "table text"
    mock_table.metadata.text_as_html = "<table></table>"

    mock_elements = [mock_title, mock_table]

    # Mock partition_pdf where it is used
    with patch("app.services.parser.partition_pdf", return_value=mock_elements):
        result = parser_service.parse_pdf(b"fake pdf")
        assert "# Doc Title" in result
        assert "[Table]" in result
        assert "<table></table>" in result


def _element(category, text, page):
    el = MagicMock()
    el.category = category
    el.text = text
    el.metadata.page_number = page
    el.metadata.text_as_html = None
    return el


def test_parse_pdf_pages_groups_by_page_and_skips_footers():
    elements = [
        _element("Title", "Handbook", 1),
        _element("NarrativeText", "Vacation is 15 days.", 1),
        _element("PageNumber", "1", 1),
        _element("NarrativeText", "Parking is behind building two.", 2),
    ]
    with patch("app.services.parser.partition_pdf", return_value=elements):
        pages = parser_service.parse_pages(b"fake pdf", "pdf")

    assert [p for p, _ in pages] == [1, 2]
    assert "Vacation is 15 days." in pages[0][1]
    assert "Parking" in pages[1][1]
    assert "Parking" not in pages[0][1]


def test_parse_pages_without_page_concept():
    assert parser_service.parse_pages(b"Hello", "txt") == [(None, "Hello")]
