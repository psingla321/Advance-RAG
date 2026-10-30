import re
from typing import Any

from app.textutil import content_terms, split_sentences

SUPPORT_THRESHOLD = 0.6
_CITATION = re.compile(r"\[\d+\]")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def claim_sentences(answer: str) -> list[str]:
    claims: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or stripped.endswith(":") or stripped.startswith("#"):
            continue
        claims.extend(split_sentences(stripped.lstrip("-*• ").strip()))
    return claims


def check_groundedness(answer: str, sources: list[str]) -> dict[str, Any]:
    """Lexical entailment proxy: share of claim terms found in the best-matching source."""
    source_terms = [content_terms(s) for s in sources]
    results = []
    for sentence in claim_sentences(answer):
        terms = content_terms(_CITATION.sub("", sentence))
        if len(terms) < 3:
            continue
        support = max((len(terms & st) / len(terms) for st in source_terms), default=0.0)
        results.append({"sentence": sentence, "support": round(support, 3), "supported": support >= SUPPORT_THRESHOLD})
    score = 1.0 if not results else sum(r["supported"] for r in results) / len(results)
    return {"groundedness": round(score, 3), "claims": results}


def repair_answer(answer: str, claims: list[dict[str, Any]]) -> str:
    """Surgically remove unsupported sentences, keeping supported ones and structure."""
    unsupported = [c["sentence"] for c in claims if not c["supported"]]
    lines = []
    for line in answer.splitlines():
        repaired = line
        for sentence in unsupported:
            repaired = repaired.replace(sentence, "")
        if line.strip() and not repaired.strip(" -*•"):
            continue
        lines.append(repaired.rstrip())
    return "\n".join(lines).strip()


def numeric_check(answer: str, rows: list[list[Any]], question: str = "") -> float:
    allowed: set[str] = set()
    for row in rows:
        for value in row:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                allowed.add(f"{float(value):.2f}")
                allowed.add(f"{round(float(value)):.2f}")
    allowed |= {f"{float(n.replace(',', '')):.2f}" for n in _NUMBER.findall(question)}
    text = _CITATION.sub("", "\n".join(l for l in answer.splitlines() if not l.strip().startswith("|")))
    numbers = [n for n in _NUMBER.findall(text) if len(n.replace(",", "").lstrip("-")) > 1]
    if not numbers:
        return 1.0
    matched = sum(1 for n in numbers if f"{float(n.replace(',', '')):.2f}" in allowed)
    return matched / len(numbers)
