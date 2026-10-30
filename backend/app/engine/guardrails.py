import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
        r"\bdisregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|your)\s+(instructions|rules|guidelines)",
        r"\b(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+prompt|hidden\s+instructions|instructions)",
        r"\byou\s+are\s+now\s+(in\s+)?(developer\s+mode|dan|jailbroken)",
        r"\bjailbreak\b",
    )
]

_PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


@dataclass
class GuardResult:
    blocked: bool
    reason: str | None = None
    pii: list[str] = field(default_factory=list)


def check_input(text: str, max_length: int = 4000) -> GuardResult:
    if len(text) > max_length:
        return GuardResult(True, "Message is too long.")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardResult(True, "The request looks like a prompt-injection attempt and was blocked by policy.")
    pii = [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]
    return GuardResult(False, None, pii)


def redact(text: str) -> str:
    for name, pattern in _PII_PATTERNS.items():
        text = pattern.sub(f"[{name.upper()}]", text)
    return text
