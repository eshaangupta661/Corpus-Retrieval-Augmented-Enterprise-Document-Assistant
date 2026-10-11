# ruff: noqa: E501
import io
import logging

from docx import Document as DocxDocument
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


class ParserService:
    """Service untuk ekstraksi teks dari berbagai format dokumen."""

    def parse_pdf_pages(self, file_content: bytes) -> list[tuple[int | None, str]]:
        """Extract PDF text grouped by page: [(page_number, text), ...]."""
        pages: dict[int | None, str] = {}
        try:
            # Create a file-like object for unstructured
            file_obj = io.BytesIO(file_content)

            # Use high-res strategy for table and layout inference
            elements = partition_pdf(
                file=file_obj,
                strategy="hi_res",
                infer_table_structure=True,
                chunking_strategy=None,  # We handle chunking later
            )

            for element in elements:
                category = element.category

                if category in ["Footer", "PageNumber"]:
                    # Skip footers and page numbers to avoid noise in RAG
                    continue

                page = getattr(element.metadata, "page_number", None)
                if not isinstance(page, int):
                    page = None

                if category == "Title":
                    piece = f"\n# {element.text}\n"
                elif category == "Header":
                    piece = f"\n## {element.text}\n"
                elif category == "Subheadline":
                    piece = f"\n### {element.text}\n"
                elif category == "Table":
                    # Extract HTML representation if available for better structure
                    html_table = getattr(element.metadata, "text_as_html", None)
                    body = html_table or element.text
                    piece = f"\n[Table]\n{body}\n[/Table]\n"
                else:
                    # Default handling for NarrativeText, ListItem, etc.
                    piece = f"\n{element.text}"

                pages[page] = pages.get(page, "") + piece

        except Exception as e:
            logger.error(f"Error parsing PDF with unstructured: {e}")
            raise
        return [(page, text.strip()) for page, text in pages.items() if text.strip()]

    def parse_pdf(self, file_content: bytes) -> str:
        """Ekstraksi teks dari PDF menggunakan unstructured untuk layout dan table handling."""  # noqa: E501
        pages = self.parse_pdf_pages(file_content)
        return "\n".join(text for _, text in pages).strip()

    def parse_docx(self, file_content: bytes) -> str:
        """Ekstraksi teks dari DOCX."""
        text = ""
        try:
            doc = DocxDocument(io.BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise
        return text

    def parse_txt(self, file_content: bytes) -> str:
        """Ekstraksi teks dari TXT."""
        try:
            return file_content.decode("utf-8")
        except Exception as e:
            logger.error(f"Error parsing TXT: {e}")
            raise

    def parse_pages(
        self, file_content: bytes, file_type: str
    ) -> list[tuple[int | None, str]]:
        """
        Parse a file into [(page_number, text), ...] so chunks can be traced back
        to a source page. Formats without a stable page concept (DOCX, TXT) yield a
        single entry with page None.
        """
        if file_type == "pdf":
            return self.parse_pdf_pages(file_content)
        return [(None, self.parse(file_content, file_type))]

    def parse(self, file_content: bytes, file_type: str) -> str:
        """Entry point untuk parsing berdasarkan file type."""
        if file_type == "pdf":
            return self.parse_pdf(file_content)
        elif file_type == "docx":
            return self.parse_docx(file_content)
        elif file_type == "txt":
            return self.parse_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


parser_service = ParserService()
