import json
import logging
import math
import re
from dataclasses import asdict, dataclass

from app.llm.client import LLMClient

log = logging.getLogger(__name__)

ROUTE_LABELS = {"A": "Direct LLM", "B": "Document RAG", "D": "SQL analytics", "E": "Tool calling", "H": "Human escalation"}


def _p(pattern: str, weight: float = 1.2) -> tuple[re.Pattern[str], float]:
    return re.compile(pattern, re.IGNORECASE), weight


KEYWORDS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "D": [
        _p(r"\brevenue\b", 1.5), _p(r"\bsales\b", 1.5), _p(r"\bunits?\b"), _p(r"\b(total|sum)\b"),
        _p(r"\b(average|avg|mean)\b"), _p(r"\btop\s+\d+\b"),
        _p(r"\b(by|per|each|across)\s+(region|product|quarter|year)s?\b", 1.5), _p(r"\bq[1-4]\b"),
        _p(r"\b20\d\d\b", 0.8), _p(r"\bhow many\b", 0.8), _p(r"\b(trend|over time|quarterly)\b"),
        _p(r"\b(highest|lowest|best|worst)\b", 0.8), _p(r"\b(emea|apac|latam|north america)\b"),
    ],
    "E": [
        _p(r"\b(calculate|compute|evaluate)\b", 2.0), _p(r"\d+(\.\d+)?\s*[-+*/^%x]\s*\(?\d", 3.5),
        _p(r"\b(plus|minus|times|divided by)\b", 1.5),
        _p(r"\b(what time|current time|time now|today'?s date|what date|what day)\b", 3.0),
        _p(r"\b(create|open|raise|file|log|submit)\b.{0,40}\bticket\b", 4.0), _p(r"\bticket\b", 1.0),
    ],
    "B": [
        _p(r"\bpolic(y|ies)\b", 1.5), _p(r"\bdocument(s|ation)?\b"), _p(r"\baccording to\b", 1.5), _p(r"\bour\b", 0.8),
        _p(r"\b(procedure|process|guideline|handbook|manual|contract|agreement|standard)s?\b"),
        _p(r"\b(company|employee|staff)\b", 0.8),
        _p(r"\b(benefit|leave|travel|expense|reimburse\w*|remote|vacation|security|password|onboarding|compensation|salary|band)s?\b"),
        _p(r"\b(uploaded|file)\b"),
    ],
    "A": [
        _p(r"^\s*(what is|what are|who is|who was|define|explain|how does|how do|why)\b", 1.0),
        _p(r"\b(write|draft|translate|rephrase|brainstorm|poem|joke|story)\b", 1.8), _p(r"\bin general\b", 1.5),
    ],
    "H": [
        _p(r"\b(human|real person|live agent|representative)\b", 2.5), _p(r"\bescalate\b", 2.5),
        _p(r"\b(talk|speak)\s+(to|with)\b", 2.0), _p(r"\bsomeone\b", 1.0),
    ],
}
PRIORS = {"A": 0.5, "B": 0.9, "D": 0.0, "E": 0.0, "H": -0.5}

RISK_PATTERN = re.compile(
    r"\b(medical|diagnos\w*|lawsuit|legal advice|terminat\w*|fired|harass\w*|discriminat\w*|salary|salaries|"
    r"compensation|hipaa|gdpr|disciplin\w*)\b",
    re.IGNORECASE,
)
GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b[\s!.]*$", re.IGNORECASE)
CONFIRM = re.compile(r"^\s*(confirm|yes|yes,?\s+(please|go ahead)|go ahead|do it)\b", re.IGNORECASE)
SLASH_COMMANDS = {"/sql": "D", "/docs": "B", "/tool": "E", "/human": "H", "/chat": "A"}

LLM_ROUTER_PROMPT = """You are the query router of an enterprise AI assistant. Score how well each execution route fits the request.
Routes:
A = answer from general knowledge (no company data needed)
B = search the company's uploaded documents (policies, handbooks, procedures)
D = query the sales database: table sales(region, product, year, quarter, revenue, units)
E = use a tool: calculator, current date/time, or create a support ticket
H = the user explicitly wants a human
Return JSON only: {"A": p, "B": p, "D": p, "E": p, "H": p} with probabilities that sum to 1."""


@dataclass
class RoutingDecision:
    mode: str
    routes: list[dict]
    execute: list[str]
    confidence: float
    risk: float
    ambiguity: float
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def heuristic_scores(query: str, has_documents: bool = True) -> dict[str, float]:
    scores = dict(PRIORS)
    for route, patterns in KEYWORDS.items():
        for pattern, weight in patterns:
            if pattern.search(query):
                scores[route] += weight
    if not has_documents:
        scores["B"] -= 1.5
    return scores


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    peak = max(scores.values())
    exp = {k: math.exp((v - peak) / temperature) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def normalized_entropy(probs: dict[str, float]) -> float:
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return entropy / math.log(len(probs))


def strip_slash_command(query: str) -> str:
    lowered = query.lower()
    for prefix in SLASH_COMMANDS:
        if lowered.startswith(prefix):
            return query[len(prefix):].strip() or query
    return query


class QueryRouter:
    def __init__(self, llm: LLMClient, single_threshold: float = 0.85, parallel_threshold: float = 0.60) -> None:
        self._llm = llm
        self._single = single_threshold
        self._parallel = parallel_threshold

    async def classify(self, query: str, *, has_documents: bool = True, pending_action: bool = False) -> RoutingDecision:
        text = query.strip()
        risk = 0.6 if RISK_PATTERN.search(text) else 0.1

        for prefix, route in SLASH_COMMANDS.items():
            if text.lower().startswith(prefix):
                return self._forced(route, risk, f"L0 rule: slash command {prefix}")
        if pending_action and CONFIRM.match(text):
            return self._forced("E", risk, "L0 rule: confirmation of a pending action")
        if GREETING.match(text):
            return self._forced("A", risk, "L0 rule: greeting")

        probs = softmax(heuristic_scores(text, has_documents))
        source = "L1 heuristic classifier"
        if not self._llm.is_mock:
            llm_probs = await self._llm_scores(text)
            if llm_probs:
                probs = {r: 0.7 * llm_probs.get(r, 0.0) + 0.3 * p for r, p in probs.items()}
                source = "L1 LLM + heuristic ensemble"
        return self.decide(probs, risk, source)

    def decide(self, probs: dict[str, float], risk: float, source: str) -> RoutingDecision:
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        (top, p1), (second, p2) = ranked[0], ranked[1]
        ambiguity = round(normalized_entropy(probs), 3)
        routes = [{"route": r, "confidence": round(p, 3)} for r, p in ranked]

        if top == "H" and p1 >= self._parallel:
            mode, execute = "escalate", []
        elif p1 >= self._single:
            mode, execute = "single", [top]
        elif p1 >= self._parallel:
            mode, execute = "parallel", [r for r in (top, second) if r != "H"]
        elif risk >= 0.5:
            mode, execute = "escalate", []
        elif p1 - p2 < 0.05 and p1 < 0.45 and "H" not in (top, second):
            mode, execute = "clarify", [top, second]
        else:
            mode, execute = "plan", [r for r, p in ranked if p >= 0.15 and r != "H"] or ["B"]

        explanation = f"{source}: {ROUTE_LABELS[top]} {p1:.0%}, {ROUTE_LABELS[second]} {p2:.0%} -> {mode}"
        return RoutingDecision(mode, routes, execute, round(p1, 3), risk, ambiguity, explanation)

    def _forced(self, route: str, risk: float, reason: str) -> RoutingDecision:
        return RoutingDecision(
            mode="escalate" if route == "H" else "single",
            routes=[{"route": route, "confidence": 1.0}],
            execute=[] if route == "H" else [route],
            confidence=1.0,
            risk=risk,
            ambiguity=0.0,
            explanation=reason,
        )

    async def _llm_scores(self, query: str) -> dict[str, float] | None:
        try:
            raw = await self._llm.complete(
                [{"role": "system", "content": LLM_ROUTER_PROMPT}, {"role": "user", "content": query}],
                json_mode=True,
                temperature=0,
                max_tokens=100,
            )
            data = json.loads(raw)
            scores = {r: max(0.0, float(data.get(r, 0.0))) for r in PRIORS}
            total = sum(scores.values())
            return {r: s / total for r, s in scores.items()} if total > 0 else None
        except Exception as exc:
            log.warning("LLM router failed, using heuristic only: %s", exc.__class__.__name__)
            return None
