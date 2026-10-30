"""Document loaders: bytes -> structured sections.

``auto`` dispatches on file extension and is what the pipeline uses by default.
Adding a format is a single registered class.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path, PurePath

from app.rag.base import Loader
from app.rag.registry import build, names, register
from app.rag.types import Section

log = logging.getLogger(__name__)


@register("loader", "text")
class TextLoader(Loader):
    """Plain text / source code: one section per file."""

    extensions = (".txt", ".log", ".py", ".sql", ".yaml", ".yml", ".ini", ".rst")

    def load(self, filename: str, data: bytes) -> list[Section]:
        return [Section(PurePath(filename).stem, data.decode("utf-8", errors="replace"))]


@register("loader", "markdown")
class MarkdownLoader(Loader):
    """Markdown with a breadcrumb heading path (``H1 > H2 > H3``) per section."""

    extensions = (".md", ".markdown", ".mdx")

    def load(self, filename: str, data: bytes) -> list[Section]:
        sections: list[Section] = []
        path: list[str] = []
        buffer: list[str] = []
        in_code = False

        def flush() -> None:
            if any(line.strip() for line in buffer):
                sections.append(Section(" > ".join(path), "\n".join(buffer).strip()))
            buffer.clear()

        for line in data.decode("utf-8", errors="replace").splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if line.startswith("#") and not in_code:
                flush()
                level = len(line) - len(line.lstrip("#"))
                path[:] = path[: level - 1] + [line.lstrip("#").strip()]
            else:
                buffer.append(line)
        flush()
        return sections


@register("loader", "html", requires=["beautifulsoup4"])
class HtmlLoader(Loader):
    """HTML with scripts/styles stripped; falls back to a regex strip if bs4 is absent."""

    extensions = (".html", ".htm", ".xhtml")

    def load(self, filename: str, data: bytes) -> list[Section]:
        text = data.decode("utf-8", errors="replace")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            sections: list[Section] = []
            heading = soup.title.get_text(strip=True) if soup.title else PurePath(filename).stem
            buffer: list[str] = []
            for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre"]):
                content = element.get_text(" ", strip=True)
                if not content:
                    continue
                if element.name.startswith("h"):
                    if buffer:
                        sections.append(Section(heading, "\n".join(buffer)))
                        buffer = []
                    heading = content
                else:
                    buffer.append(content)
            if buffer:
                sections.append(Section(heading, "\n".join(buffer)))
            return sections or [Section(PurePath(filename).stem, soup.get_text(" ", strip=True))]
        except ImportError:
            stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
            return [Section(PurePath(filename).stem, re.sub(r"<[^>]+>", " ", stripped))]


@register("loader", "pdf", requires=["pypdf"])
class PdfLoader(Loader):
    """PDF text per page. Set ``ocr=True`` to fall back to OCR on image-only pages."""

    extensions = (".pdf",)

    def __init__(self, ocr: bool = False, min_chars_for_ocr: int = 40) -> None:
        self._ocr = ocr
        self._min_chars = min_chars_for_ocr

    def load(self, filename: str, data: bytes) -> list[Section]:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        sections: list[Section] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if self._ocr and len(text.strip()) < self._min_chars:
                text = self._ocr_page(data, i) or text
            sections.append(Section(f"Page {i}", text, {"page": i}))
        return sections

    def _ocr_page(self, data: bytes, page: int) -> str:
        try:
            import pytesseract
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(data, first_page=page, last_page=page)
            return "\n".join(pytesseract.image_to_string(image) for image in images)
        except Exception as exc:  # OCR is best-effort
            log.warning("OCR unavailable for page %s: %s", page, exc)
            return ""


@register("loader", "docx", requires=["python-docx"])
class DocxLoader(Loader):
    """Word documents: heading-aware sections plus one section per table."""

    extensions = (".docx",)

    def load(self, filename: str, data: bytes) -> list[Section]:
        from docx import Document as DocxDocument

        document = DocxDocument(io.BytesIO(data))
        sections: list[Section] = []
        heading, buffer = PurePath(filename).stem, []
        for paragraph in document.paragraphs:
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            if style.lower().startswith("heading") and paragraph.text.strip():
                if buffer:
                    sections.append(Section(heading, "\n".join(buffer)))
                heading, buffer = paragraph.text.strip(), []
            elif paragraph.text.strip():
                buffer.append(paragraph.text.strip())
        if buffer:
            sections.append(Section(heading, "\n".join(buffer)))
        for index, table in enumerate(document.tables, 1):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            sections.append(Section(f"Table {index}", "\n".join(rows), {"kind": "table"}))
        return sections


@register("loader", "csv")
class CsvLoader(Loader):
    """Tabular data rendered as ``column: value`` sentences so retrieval can match fields."""

    extensions = (".csv", ".tsv")

    def __init__(self, rows_per_section: int = 25) -> None:
        self._rows = rows_per_section

    def load(self, filename: str, data: bytes) -> list[Section]:
        text = data.decode("utf-8", errors="replace")
        delimiter = "\t" if filename.lower().endswith(".tsv") else ","
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        if not rows:
            return []
        header, body = rows[0], rows[1:]
        sections = []
        for start in range(0, len(body), self._rows):
            lines = [
                "; ".join(f"{h}: {v}" for h, v in zip(header, row)) + "."
                for row in body[start : start + self._rows]
            ]
            sections.append(Section(f"Rows {start + 1}-{start + len(lines)}", "\n".join(lines), {"kind": "table"}))
        return sections


@register("loader", "json")
class JsonLoader(Loader):
    """JSON / JSONL: one section per top-level key, array item or line."""

    extensions = (".json", ".jsonl", ".ndjson")

    def load(self, filename: str, data: bytes) -> list[Section]:
        text = data.decode("utf-8", errors="replace")
        if filename.lower().endswith((".jsonl", ".ndjson")):
            return [
                Section(f"Record {i}", json.dumps(json.loads(line), indent=1, ensure_ascii=False))
                for i, line in enumerate((l for l in text.splitlines() if l.strip()), 1)
            ]
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [Section(str(k), json.dumps(v, indent=1, ensure_ascii=False)) for k, v in parsed.items()]
        if isinstance(parsed, list):
            return [Section(f"Item {i}", json.dumps(v, indent=1, ensure_ascii=False)) for i, v in enumerate(parsed, 1)]
        return [Section(PurePath(filename).stem, str(parsed))]


@register("loader", "pptx", requires=["python-pptx"])
class PptxLoader(Loader):
    """PowerPoint: one section per slide, including speaker notes."""

    extensions = (".pptx",)

    def load(self, filename: str, data: bytes) -> list[Section]:
        from pptx import Presentation

        presentation = Presentation(io.BytesIO(data))
        sections = []
        for i, slide in enumerate(presentation.slides, 1):
            parts = [shape.text.strip() for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                parts.append("Notes: " + slide.notes_slide.notes_text_frame.text.strip())
            if parts:
                sections.append(Section(f"Slide {i}", "\n".join(parts), {"slide": i}))
        return sections


@register("loader", "xlsx", requires=["openpyxl"])
class XlsxLoader(Loader):
    """Excel: one section per worksheet, rows rendered as ``column: value``."""

    extensions = (".xlsx", ".xlsm")

    def __init__(self, rows_per_section: int = 25) -> None:
        self._rows = rows_per_section

    def load(self, filename: str, data: bytes) -> list[Section]:
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sections: list[Section] = []
        for sheet in workbook.worksheets:
            rows = [[("" if c is None else str(c)) for c in row] for row in sheet.iter_rows(values_only=True)]
            if not rows:
                continue
            header, body = rows[0], rows[1:]
            for start in range(0, len(body), self._rows):
                lines = ["; ".join(f"{h}: {v}" for h, v in zip(header, row) if v) + "." for row in body[start : start + self._rows]]
                sections.append(Section(f"{sheet.title} rows {start + 1}-{start + len(lines)}", "\n".join(lines), {"sheet": sheet.title}))
        return sections


@register("loader", "auto")
class AutoLoader(Loader):
    """Dispatches on file extension to every other registered loader."""

    def __init__(self, pdf_ocr: bool = False) -> None:
        self._pdf_ocr = pdf_ocr
        self._cache: dict[str, Loader] = {}

    def _for(self, extension: str) -> Loader:
        from app.rag.registry import get_class

        for name in names("loader"):
            if name == "auto":
                continue
            cls = get_class("loader", name)
            if extension in getattr(cls, "extensions", ()):
                if name not in self._cache:
                    self._cache[name] = build("loader", {"type": name, **({"ocr": self._pdf_ocr} if name == "pdf" else {})})
                return self._cache[name]
        self._cache.setdefault("text", build("loader", "text"))
        return self._cache["text"]

    def load(self, filename: str, data: bytes) -> list[Section]:
        return self._for(PurePath(filename).suffix.lower()).load(filename, data)


SUPPORTED_EXTENSIONS = {
    ".txt", ".log", ".py", ".sql", ".yaml", ".yml", ".ini", ".rst", ".md", ".markdown", ".mdx",
    ".html", ".htm", ".xhtml", ".pdf", ".docx", ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
    ".pptx", ".xlsx", ".xlsm",
}


def iter_files(path: str | Path, recursive: bool = True, extensions: set[str] | None = None) -> list[Path]:
    """Walk a directory (or accept a single file) and return loadable paths."""
    root = Path(path)
    allowed = extensions or SUPPORTED_EXTENSIONS
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in allowed and not any(part.startswith(".") for part in p.parts)
    )
