import re
from pathlib import PurePath

ALLOWED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md", ".csv", ".json"})
_EXECUTABLE_MAGIC = (b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"#!")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


class UploadRejected(ValueError):
    pass


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Validate type, size and content signature; return a sanitized filename."""
    name = _SAFE_NAME.sub("_", PurePath(filename.replace("\\", "/")).name).strip(" .")[:150]
    extension = PurePath(name).suffix.lower()
    if not name or extension not in ALLOWED_EXTENSIONS:
        raise UploadRejected(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if not data:
        raise UploadRejected("The file is empty.")
    if len(data) > max_bytes:
        raise UploadRejected("The file exceeds the maximum upload size.")
    if data.startswith(_EXECUTABLE_MAGIC):
        raise UploadRejected("Executable content is not allowed.")
    if extension == ".pdf" and not data.startswith(b"%PDF"):
        raise UploadRejected("The file is not a valid PDF.")
    if extension == ".docx" and not data.startswith(b"PK"):
        raise UploadRejected("The file is not a valid DOCX document.")
    if extension in {".txt", ".md", ".csv", ".json"}:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadRejected("Text files must be UTF-8 encoded.") from exc
    return name
