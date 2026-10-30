import hashlib
import math
import re

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our ours please show tell that the their them then there these they this those to us was we
    were what when where which who whom why will with would you your about give list find get""".split()
)
NEGATIONS = frozenset({"not", "no", "without", "except", "excluding", "never"})


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_term_list(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    return set(content_term_list(text))


def keyword_terms(text: str, limit: int = 20) -> list[str]:
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if token not in STOPWORDS and len(token) > 1:
            seen.setdefault(token, None)
    return list(seen)[:limit]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def normalize_query(query: str) -> str:
    return " ".join(tokenize(query))


def sha256_hex(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def acl_hash(groups: list[str]) -> str:
    return sha256_hex(",".join(sorted(set(groups))))[:16]


def entity_signature(query: str) -> frozenset[str]:
    """Numbers, quarters, negations and mid-sentence proper nouns that must match for a semantic cache hit."""
    signature = set(_NUMBER.findall(query.replace(",", "")))
    signature |= {t for t in tokenize(query) if re.fullmatch(r"q[1-4]", t) or t in NEGATIONS}
    for i, raw in enumerate(query.split()):
        word = raw.strip(".,?!:;\"'()")
        if i > 0 and word[:1].isupper():
            signature.add(word.lower())
    return frozenset(signature)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic feature-hashing embedding used when no embedding model is configured."""
    vector = [0.0] * dim
    terms = content_term_list(text)
    features = [(t, 1.0) for t in terms] + [(f"{a}_{b}", 0.5) for a, b in zip(terms, terms[1:])]
    for feature, weight in features:
        h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
        vector[h % dim] += weight if (h >> 63) & 1 else -weight
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
