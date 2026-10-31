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

    extensions = (
        ".txt", ".log", ".py", ".sql", ".yaml", ".yml", ".ini", ".rst",
        ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
        ".rb", ".php", ".sh", ".bash", ".ps1", ".toml", ".cfg", ".tf", ".kt", ".swift", ".scala",
    )

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


@register("loader", "notebook")
class NotebookLoader(Loader):
    """Jupyter notebooks: one section per cell, code kept in fenced blocks.

    Outputs are dropped — they are usually noise, and often huge.
    """

    extensions = (".ipynb",)

    def load(self, filename: str, data: bytes) -> list[Section]:
        try:
            document = json.loads(data.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{filename} is not valid notebook JSON: {exc}") from exc
        language = (
            document.get("metadata", {}).get("language_info", {}).get("name")
            or document.get("metadata", {}).get("kernelspec", {}).get("language")
            or "python"
        )
        stem = PurePath(filename).stem
        sections: list[Section] = []
        for index, cell in enumerate(document.get("cells", []), 1):
            source = cell.get("source", "")
            text = "".join(source) if isinstance(source, list) else str(source)
            if not text.strip():
                continue
            kind = cell.get("cell_type", "code")
            body = f"```{language}\n{text.strip()}\n```" if kind == "code" else text.strip()
            sections.append(Section(f"{stem} > cell {index} ({kind})", body))
        return sections


@register("loader", "email")
class EmailLoader(Loader):
    """RFC 822 email (``.eml``): headers as a section, then each text part.

    HTML-only mail is reduced to text by the same tag stripper the EPUB loader uses.
    """

    extensions = (".eml", ".mbox")

    def load(self, filename: str, data: bytes) -> list[Section]:
        from email import policy
        from email.parser import BytesParser

        message = BytesParser(policy=policy.default).parsebytes(data)
        subject = str(message.get("subject", "") or PurePath(filename).stem)
        header_lines = [
            f"{field}: {message.get(field)}"
            for field in ("From", "To", "Cc", "Date", "Subject")
            if message.get(field)
        ]
        sections = [Section(subject, "\n".join(header_lines))] if header_lines else []

        for part in message.walk() if message.is_multipart() else [message]:
            content_type = part.get_content_type()
            if content_type not in ("text/plain", "text/html") or part.get_filename():
                continue
            try:
                body = part.get_content()
            except (LookupError, ValueError):  # unknown charset or malformed part
                payload = part.get_payload(decode=True) or b""
                body = payload.decode("utf-8", errors="replace")
            if content_type == "text/html":
                body = strip_tags(body)
            if body.strip():
                sections.append(Section(subject, body.strip()))
        return sections or [Section(subject, data.decode("utf-8", errors="replace"))]


# Entity declarations are the lever behind billion-laughs and XXE, so documents
# that carry one are refused rather than parsed.
_DOCTYPE = re.compile(rb"<!DOCTYPE[^>]*(\[|ENTITY)", re.I)
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_WHITESPACE = re.compile(r"[ \t]*\n\s*\n\s*")


def strip_tags(markup: str) -> str:
    """Reduce markup to readable text without needing an HTML parser installed.

    Block elements become blank lines so the structure a chunker relies on survives.
    """
    from html import unescape

    text = _SCRIPT_OR_STYLE.sub(" ", markup)
    text = re.sub(r"<(?:br|/li|/td)\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(?:p|div|h\d|tr|section|article|blockquote|table|ul|ol)\s*>", "\n\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)  # no stray spaces hugging a line break
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@register("loader", "xml")
class XmlLoader(Loader):
    """Generic XML: one section per repeated child element, rendered as ``tag: value``.

    Refuses documents containing entity declarations.
    """

    extensions = (".xml",)

    def __init__(self, rows_per_section: int = 25) -> None:
        self._rows = rows_per_section

    def load(self, filename: str, data: bytes) -> list[Section]:
        import xml.etree.ElementTree as ElementTree

        if _DOCTYPE.search(data[:4096]):
            raise ValueError(f"{filename} declares XML entities, which are not parsed for safety.")
        try:
            root = ElementTree.fromstring(data.decode("utf-8", errors="replace"))
        except ElementTree.ParseError as exc:
            raise ValueError(f"{filename} is not well-formed XML: {exc}") from exc

        def render(element: Any, depth: int = 0) -> list[str]:
            lines: list[str] = []
            tag = str(element.tag).rsplit("}", 1)[-1]
            value = (element.text or "").strip()
            if value:
                lines.append(f"{'  ' * depth}{tag}: {value}")
            elif len(element):
                lines.append(f"{'  ' * depth}{tag}:")
            for attribute, attribute_value in element.attrib.items():
                lines.append(f"{'  ' * (depth + 1)}@{attribute}: {attribute_value}")
            for child in element:
                lines.extend(render(child, depth + 1))
            return lines

        children = list(root)
        stem = PurePath(filename).stem
        if not children:
            return [Section(stem, "\n".join(render(root)))]
        sections: list[Section] = []
        for start in range(0, len(children), self._rows):
            block = children[start : start + self._rows]
            body = "\n".join(line for child in block for line in render(child))
            if body.strip():
                sections.append(Section(f"{stem} > rows {start + 1}-{start + len(block)}", body))
        return sections


@register("loader", "epub")
class EpubLoader(Loader):
    """EPUB e-books: one section per XHTML chapter, in spine order where available."""

    extensions = (".epub",)
    # Guards against a zip bomb: a book that expands past these limits is refused.
    MAX_MEMBERS = 2000
    MAX_BYTES = 80 * 1024 * 1024

    def load(self, filename: str, data: bytes) -> list[Section]:
        import zipfile

        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{filename} is not a readable EPUB: {exc}") from exc
        with archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith((".xhtml", ".html", ".htm"))]
            if len(archive.infolist()) > self.MAX_MEMBERS:
                raise ValueError(f"{filename} has too many entries to load safely.")
            if sum(m.file_size for m in members) > self.MAX_BYTES:
                raise ValueError(f"{filename} expands to more than {self.MAX_BYTES // 1024 // 1024} MB.")

            sections: list[Section] = []
            for member in sorted(members, key=lambda m: m.filename):
                try:
                    markup = archive.read(member).decode("utf-8", errors="replace")
                except (zipfile.BadZipFile, OSError) as exc:
                    log.warning("Skipping %s in %s: %s", member.filename, filename, exc)
                    continue
                title = re.search(r"<title[^>]*>(.*?)</title>", markup, re.I | re.S)
                heading = (title.group(1).strip() if title else PurePath(member.filename).stem)[:120]
                body = strip_tags(markup)
                if body:
                    sections.append(Section(heading, body))
        return sections


@register("loader", "rtf")
class RtfLoader(Loader):
    """Rich Text Format: control words stripped, escapes and unicode decoded."""

    extensions = (".rtf",)

    def load(self, filename: str, data: bytes) -> list[Section]:
        text = data.decode("latin-1", errors="replace")
        # Drop embedded binary/metadata groups whose content is never prose.
        text = re.sub(
            r"\{\\(?:\*\\)?(?:fonttbl|colortbl|stylesheet|info|pict|object)\b[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            " ", text,
        )
        text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]).decode("cp1252", "replace"), text)
        text = re.sub(r"\\u(-?\d+)\s?\??", lambda m: chr(int(m.group(1)) % 65536), text)
        text = re.sub(r"\\(par|line|page)\b\s?", "\n", text)
        text = re.sub(r"\\tab\b\s?", "\t", text)
        text = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", text)   # remaining control words
        text = text.replace("{", " ").replace("}", " ")
        text = re.sub(r"[ \t]{2,}", " ", text)
        body = _WHITESPACE.sub("\n\n", text).strip()
        return [Section(PurePath(filename).stem, body)] if body else []


@register("loader", "parquet", requires=["pyarrow"])
class ParquetLoader(Loader):
    """Columnar Parquet, rendered as ``column: value`` sentences like the CSV loader."""

    extensions = (".parquet",)

    def __init__(self, rows_per_section: int = 25, max_rows: int = 10_000) -> None:
        self._rows = rows_per_section
        self._max_rows = max_rows

    def load(self, filename: str, data: bytes) -> list[Section]:
        import pyarrow.parquet as parquet

        table = parquet.read_table(io.BytesIO(data))
        columns = table.column_names
        records = table.slice(0, self._max_rows).to_pylist()
        stem = PurePath(filename).stem
        sections: list[Section] = []
        for start in range(0, len(records), self._rows):
            block = records[start : start + self._rows]
            body = "\n".join(
                "; ".join(f"{column}: {row[column]}" for column in columns if row.get(column) is not None)
                for row in block
            )
            if body.strip():
                sections.append(Section(f"{stem} > rows {start + 1}-{start + len(block)}", body))
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
    # Added alongside the loaders below; keep in step with each class's `extensions`.
    ".ipynb", ".eml", ".mbox", ".xml", ".epub", ".rtf", ".parquet",
    ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".rb", ".php", ".sh", ".bash", ".ps1", ".toml", ".cfg", ".tf", ".kt", ".swift", ".scala",
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
