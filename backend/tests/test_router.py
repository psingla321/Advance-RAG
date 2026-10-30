import pytest

from app.engine.router import QueryRouter, heuristic_scores, softmax
from app.llm.client import MockLLM


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter(MockLLM(64))


async def _classify(router: QueryRouter, query: str, **kwargs):
    return await router.classify(query, **kwargs)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("What is our travel reimbursement policy?", "B"),
        ("Total revenue by region in 2025", "D"),
        ("What is 25 * 4?", "E"),
        ("I want to talk to a human", "H"),
        ("Create a P2 ticket for VPN outage", "E"),
        ("Write a short poem about clouds", "A"),
    ],
)
def test_top_route(router: QueryRouter, query: str, expected: str) -> None:
    import asyncio

    decision = asyncio.run(_classify(router, query))
    assert decision.routes[0]["route"] == expected


def test_single_route_for_confident_query(router: QueryRouter) -> None:
    import asyncio

    decision = asyncio.run(_classify(router, "Total revenue by region in 2025"))
    assert decision.mode == "single"
    assert decision.execute == ["D"]


def test_human_request_escalates(router: QueryRouter) -> None:
    import asyncio

    assert asyncio.run(_classify(router, "Please escalate, I need a human")).mode == "escalate"


def test_slash_command_forces_route(router: QueryRouter) -> None:
    import asyncio

    decision = asyncio.run(_classify(router, "/sql revenue"))
    assert decision.execute == ["D"] and decision.confidence == 1.0


def test_confirm_requires_pending_action(router: QueryRouter) -> None:
    import asyncio

    assert asyncio.run(_classify(router, "confirm", pending_action=True)).execute == ["E"]
    assert asyncio.run(_classify(router, "confirm", pending_action=False)).execute != ["E"]


def test_no_documents_penalizes_rag() -> None:
    with_docs = softmax(heuristic_scores("What is the policy?", True))["B"]
    without_docs = softmax(heuristic_scores("What is the policy?", False))["B"]
    assert without_docs < with_docs
