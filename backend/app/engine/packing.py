from typing import Any

from app.textutil import content_terms, estimate_tokens, jaccard


def pack_context(
    candidates: list[dict[str, Any]],
    query: str,
    token_budget: int,
    coverage_weight: float = 0.5,
    redundancy_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """Greedy cost-benefit selection maximising relevance + query-aspect coverage - redundancy under a token budget."""
    query_terms = content_terms(query)
    pool = [
        (c, content_terms(f"{c.get('heading', '')} {c['content']}"), estimate_tokens(c["content"]))
        for c in candidates
    ]
    selected: list[int] = []
    covered: set[str] = set()
    used = 0

    while True:
        best_index, best_ratio, best_gain = -1, 0.0, 0.0
        for i, (candidate, terms, tokens) in enumerate(pool):
            if i in selected or used + tokens > token_budget:
                continue
            new_terms = (terms & query_terms) - covered
            coverage_gain = len(new_terms) / max(1, len(query_terms))
            redundancy = max((jaccard(terms, pool[j][1]) for j in selected), default=0.0)
            gain = candidate.get("score", 0.0) + coverage_weight * coverage_gain - redundancy_weight * redundancy
            ratio = gain / tokens
            if gain > 0 and ratio > best_ratio:
                best_index, best_ratio, best_gain = i, ratio, gain
        if best_index < 0:
            break
        selected.append(best_index)
        covered |= pool[best_index][1] & query_terms
        used += pool[best_index][2]

    selected.sort(key=lambda i: pool[i][0].get("score", 0.0), reverse=True)
    return [pool[i][0] for i in selected]
