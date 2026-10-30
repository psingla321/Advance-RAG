import json
import logging
import operator
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from app.config import Settings
from app.engine import generation as gen
from app.engine.guardrails import check_input
from app.engine.retrieval import HybridRetriever
from app.engine.router import CONFIRM, ROUTE_LABELS, QueryRouter, strip_slash_command
from app.engine.sql_agent import SqlAgent
from app.engine.tools import ToolAgent
from app.engine.verification import check_groundedness, numeric_check, repair_answer
from app.llm.client import LLMClient
from app.observability import CACHE_LOOKUPS, CHAT_LATENCY, CHAT_REQUESTS, GROUNDEDNESS, NODE_LATENCY, TRUST, tracer
from app.services.answer_cache import AnswerCache
from app.services.db import Database
from app.services.embedder import CachedEmbedder
from app.services.insights import InsightsStore
from app.services.memory import ConversationMemory
from app.services.vector_store import VectorStore

log = logging.getLogger(__name__)

BRANCH_NODES = {"A": "direct", "B": "rag", "D": "sql", "E": "tool"}
FOLLOW_UP = re.compile(r"\b(it|its|that|this|those|these|they|them|there)\b|^\s*(and|what about|how about)\b", re.IGNORECASE)
BLOCKED_PREFIX = "I can't help with that request."


class EngineState(TypedDict, total=False):
    request_id: str
    started_at: float
    query: str
    tenant_id: str
    groups: list[str]
    conversation_id: str
    standalone_query: str
    history: list[dict[str, Any]]
    is_followup: bool
    blocked: dict[str, Any] | None
    cache_hit: dict[str, Any] | None
    routing: dict[str, Any]
    evidence: list[dict[str, Any]]
    sql_result: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    escalation: dict[str, Any] | None
    clarification: str | None
    answer: str
    abstained: bool
    verification: dict[str, Any]
    final: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], operator.add]


def _trace(node: str, start: float, **detail: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - start) * 1000, 1), **detail}]


class Engine:
    """Agentic orchestration graph: guard -> memory -> cache -> router -> agents (parallel) -> synthesize -> verify."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        redis: Redis,
        db: Database,
        vector_store: VectorStore,
        embedder: CachedEmbedder,
        cache: AnswerCache,
        insights: InsightsStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.redis = redis
        self.db = db
        self.memory = ConversationMemory(redis)
        self.cache = cache
        self.insights = insights or InsightsStore(redis)
        self.router = QueryRouter(llm, settings.router_single_threshold, settings.router_parallel_threshold)
        self.retriever = HybridRetriever(db, vector_store, embedder, settings)
        self.sql_agent = SqlAgent(llm, db)
        self.tool_agent = ToolAgent(llm, redis)
        self.graph = self._build()

    def _build(self):
        g = StateGraph(EngineState)
        for name, fn in (
            ("guard", self._guard), ("contextualize", self._contextualize), ("cache", self._cache),
            ("route", self._route), ("direct", self._direct), ("rag", self._rag), ("sql", self._sql),
            ("tool", self._tool), ("synthesize", self._synthesize), ("verify", self._verify),
            ("escalate", self._escalate), ("clarify", self._clarify), ("finalize", self._finalize),
        ):
            g.add_node(name, self._instrument(name, fn))
        g.add_edge(START, "guard")
        g.add_conditional_edges("guard", lambda s: "finalize" if s.get("blocked") else "contextualize", ["finalize", "contextualize"])
        g.add_edge("contextualize", "cache")
        g.add_conditional_edges("cache", lambda s: "finalize" if s.get("cache_hit") else "route", ["finalize", "route"])
        g.add_conditional_edges("route", self._select_branches, [*BRANCH_NODES.values(), "escalate", "clarify"])
        for node in BRANCH_NODES.values():
            g.add_edge(node, "synthesize")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self._after_verify, ["escalate", "finalize"])
        g.add_edge("escalate", "finalize")
        g.add_edge("clarify", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    @staticmethod
    def _instrument(name: str, fn):
        async def node(state: EngineState) -> dict[str, Any]:
            with tracer.start_as_current_span(f"engine.{name}") as span:
                span.set_attribute("nexus.tenant", state.get("tenant_id", ""))
                span.set_attribute("nexus.request_id", state.get("request_id", ""))
                started = time.perf_counter()
                try:
                    return await fn(state)
                finally:
                    NODE_LATENCY.labels(name).observe(time.perf_counter() - started)

        return node

    # ---------------------------------------------------------------- nodes

    async def _guard(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        result = check_input(state["query"])
        update: dict[str, Any] = {"trace": _trace("guardrails", start, blocked=result.blocked, pii=result.pii)}
        if result.blocked:
            update["blocked"] = {"reason": result.reason}
        return update

    async def _contextualize(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        history = await self.memory.history(state["tenant_id"], state["conversation_id"], 12)
        query = state["query"]
        is_followup = bool(history) and bool(FOLLOW_UP.search(query)) and not CONFIRM.match(query)
        standalone = query
        if is_followup:
            last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            if self.llm.is_mock:
                standalone = f"{last_user} {query}"
            else:
                standalone = await self.llm.complete(
                    [
                        {"role": "system", "content": "Rewrite the follow-up as a standalone question. Return only the question."},
                        *[{"role": m["role"], "content": m["content"]} for m in history[-6:]],
                        {"role": "user", "content": query},
                    ],
                    temperature=0,
                    max_tokens=120,
                ) or query
        return {
            "history": history,
            "is_followup": is_followup,
            "standalone_query": standalone.strip(),
            "trace": _trace("memory", start, history_turns=len(history), followup=is_followup, standalone=standalone[:200]),
        }

    async def _cache(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        query = state["standalone_query"]
        if state.get("is_followup") or CONFIRM.match(query) or query.startswith("/"):
            CACHE_LOOKUPS.labels("bypass").inc()
            return {"cache_hit": None, "trace": _trace("cache", start, status="bypass")}
        hit = await self.cache.lookup(query, state["tenant_id"], state["groups"])
        status = hit["level"] if hit else "miss"
        CACHE_LOOKUPS.labels(status).inc()
        return {"cache_hit": hit, "trace": _trace("cache", start, status=status, similarity=hit.get("similarity") if hit else None)}

    async def _route(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        tenant, groups = state["tenant_id"], state["groups"]
        has_documents = await self.db.chunk_count(tenant, groups) > 0
        pending = bool(await self.redis.exists(ToolAgent.pending_key(tenant, state["conversation_id"])))
        decision = await self.router.classify(state["standalone_query"], has_documents=has_documents, pending_action=pending)
        routing = decision.to_dict()
        return {
            "routing": routing,
            "standalone_query": strip_slash_command(state["standalone_query"]),
            "trace": _trace("router", start, mode=decision.mode, execute=decision.execute, confidence=decision.confidence,
                            risk=decision.risk, explanation=decision.explanation),
        }

    def _select_branches(self, state: EngineState) -> list[str]:
        routing = state["routing"]
        if routing["mode"] == "escalate":
            return ["escalate"]
        if routing["mode"] == "clarify":
            return ["clarify"]
        return [BRANCH_NODES[r] for r in routing["execute"] if r in BRANCH_NODES] or ["rag"]

    async def _direct(self, state: EngineState) -> dict[str, Any]:
        return {"trace": _trace("direct_llm_agent", time.perf_counter(), note="answer generated at synthesis")}

    async def _rag(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        evidence, stats = await self.retriever.retrieve(state["standalone_query"], state["tenant_id"], state["groups"])
        return {"evidence": evidence, "trace": _trace("retrieval_agent", start, **stats)}

    async def _sql(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        result = await self.sql_agent.run(state["standalone_query"], state["tenant_id"])
        return {"sql_result": result, "trace": _trace("sql_agent", start, sql=result.get("sql"), rows=result.get("row_count", 0), error=result.get("error"))}

    async def _tool(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        result = await self.tool_agent.run(state["query"], state["tenant_id"], state["conversation_id"])
        return {"tool_result": result, "trace": _trace("tool_agent", start, tool=result["tool"], status=result["status"])}

    async def _synthesize(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        writer = get_stream_writer()
        query = state["standalone_query"]
        history = state.get("history", [])
        execute = state["routing"]["execute"]
        evidence = state.get("evidence") or []
        sql_result = state.get("sql_result")
        tool_result = state.get("tool_result")
        parts: list[str] = []

        def emit(text: str) -> None:
            writer({"type": "token", "text": text})

        async def emit_static(text: str) -> None:
            if parts:
                emit("\n\n")
            parts.append(text)
            for i, word in enumerate(text.split(" ")):
                emit(word if i == 0 else " " + word)

        async def emit_stream(messages: list[dict[str, str]], suffix: str = "") -> None:
            if parts:
                emit("\n\n")
            buffer: list[str] = []
            async for delta in self.llm.stream(messages):
                buffer.append(delta)
                emit(delta)
            text = "".join(buffer).strip()
            if suffix:
                emit("\n\n" + suffix)
                text = f"{text}\n\n{suffix}"
            parts.append(text)

        if tool_result:
            await emit_static(tool_result["output"])

        if sql_result:
            table = gen.render_table(sql_result.get("columns", []), sql_result.get("rows", []))
            single_cell = len(sql_result.get("rows", [])) == 1 and len(sql_result.get("columns", [])) == 1
            if sql_result.get("error") or self.llm.is_mock or not sql_result.get("rows"):
                summary = gen.sql_summary(sql_result)
                await emit_static(summary if single_cell or not table or sql_result.get("error") else f"{summary}\n\n{table}")
            else:
                messages = [
                    {"role": "system", "content": gen.SQL_SUMMARY_SYSTEM},
                    {"role": "user", "content": f"Question: {query}\nColumns: {sql_result['columns']}\nRows: {json.dumps(sql_result['rows'][:50])}"},
                ]
                await emit_stream(messages, suffix=table)

        abstained = False
        ranking = [r["route"] for r in state["routing"]["routes"]]
        prefers_documents = "B" in execute and ("A" not in ranking or ranking.index("B") < ranking.index("A"))
        if evidence:
            if self.llm.is_mock:
                await emit_static(gen.extractive_answer(query, evidence))
            else:
                await emit_stream(gen.grounded_messages(query, history, evidence))
        elif "A" in execute and not (sql_result or tool_result) and not prefers_documents:
            if self.llm.is_mock:
                await emit_static(await self.llm.complete([]))
            else:
                await emit_stream(gen.general_messages(query, history))
        elif "B" in execute and not (sql_result or tool_result):
            abstained = True
            await emit_static(gen.ABSTAIN_MESSAGE)

        if not parts:
            abstained = True
            await emit_static(gen.ABSTAIN_MESSAGE)

        answer = "\n\n".join(parts)
        return {"answer": answer, "abstained": abstained, "trace": _trace("synthesis_agent", start, chars=len(answer), abstained=abstained)}

    async def _verify(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        writer = get_stream_writer()
        answer = state["answer"]
        evidence = state.get("evidence") or []
        sql_result = state.get("sql_result")
        tool_result = state.get("tool_result")
        abstained = state.get("abstained", False)
        verification: dict[str, Any] = {"method": "none", "groundedness": None, "numeric": None, "passed": True, "repaired": False}

        if sql_result and sql_result.get("rows"):
            ratio = numeric_check(answer, sql_result["rows"], state["standalone_query"])
            verification.update(method="numeric", numeric=round(ratio, 3), passed=ratio >= 0.9)

        if evidence and not abstained:
            sources = [f"{e['heading']} {e['content']}" for e in evidence]
            if sql_result:
                sources.append(gen.rows_as_text(sql_result))
            if tool_result:
                sources.append(tool_result["output"])
            result = check_groundedness(answer, sources)
            passed = result["groundedness"] >= self.settings.groundedness_threshold
            verification.update(method="groundedness", groundedness=result["groundedness"], passed=passed and verification["passed"])
            if not passed:
                repaired = repair_answer(answer, result["claims"])
                recheck = check_groundedness(repaired, sources) if repaired else {"groundedness": 0.0, "claims": []}
                if repaired and any(c["supported"] for c in recheck["claims"]):
                    answer = repaired
                    verification.update(groundedness=recheck["groundedness"], repaired=True,
                                        passed=recheck["groundedness"] >= self.settings.groundedness_threshold)
                else:
                    answer, abstained = gen.ABSTAIN_MESSAGE, True
                    verification.update(passed=False)
                writer({"type": "revision", "answer": answer})

        return {
            "answer": answer,
            "abstained": abstained,
            "verification": verification,
            "trace": _trace("verification_agent", start, **{k: v for k, v in verification.items()}),
        }

    def _after_verify(self, state: EngineState) -> str:
        if not state["verification"]["passed"] and state["routing"]["risk"] >= 0.5:
            return "escalate"
        return "finalize"

    async def _escalate(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        if state.get("verification") and not state["verification"]["passed"]:
            reason = "Answer could not be verified for a high-risk request"
        elif state["routing"]["risk"] >= 0.5 and state["routing"]["routes"][0]["route"] != "H":
            reason = "High-risk request with low routing confidence"
        else:
            reason = "User asked for a human"
        item = {
            "id": f"HITL-{uuid4().hex[:6].upper()}",
            "query": state["query"],
            "reason": reason,
            "draft": state.get("answer"),
            "conversation_id": state["conversation_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.redis.lpush(f"hitl:{state['tenant_id']}", json.dumps(item))
        await self.redis.ltrim(f"hitl:{state['tenant_id']}", 0, 199)
        answer = (
            f"I've escalated this to a human specialist (reference **{item['id']}**). "
            f"Reason: {reason.lower()}. You'll be notified when they respond."
        )
        return {"escalation": item, "answer": answer, "trace": _trace("human_review_agent", start, ticket=item["id"], reason=reason)}

    async def _clarify(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        options = [ROUTE_LABELS[r] for r in state["routing"]["execute"]]
        text = (
            "Your question could be answered in more than one way. Did you mean:\n"
            + "\n".join(f"- {opt}" for opt in options)
            + "\nYou can also prefix your message with /docs, /sql, /tool or /chat."
        )
        return {"clarification": text, "answer": text, "trace": _trace("clarification", start, options=options)}

    async def _finalize(self, state: EngineState) -> dict[str, Any]:
        start = time.perf_counter()
        tenant, groups, conversation_id = state["tenant_id"], state["groups"], state["conversation_id"]
        cache_hit = state.get("cache_hit")
        blocked = state.get("blocked")

        if blocked:
            final = {"answer": f"{BLOCKED_PREFIX} {blocked['reason']}", "routing": None, "citations": [],
                     "verification": None, "trust": None, "blocked": True}
        elif cache_hit:
            final = {k: v for k, v in cache_hit.items() if k not in {"level", "similarity"}}
            final["cache"] = {"level": cache_hit["level"], "similarity": cache_hit["similarity"]}
        else:
            evidence = state.get("evidence") or []
            routing = state.get("routing")
            verification = state.get("verification")
            final = {
                "answer": state.get("answer", ""),
                "routing": routing,
                "citations": [
                    {"n": e["n"], "filename": e["filename"], "heading": e["heading"], "snippet": e["content"][:320],
                     "doc_id": e["doc_id"], "score": e["score"]}
                    for e in evidence
                ] if not state.get("abstained") else [],
                "sql": state.get("sql_result"),
                "tool": state.get("tool_result"),
                "verification": verification,
                "escalation": state.get("escalation"),
                "trust": self._trust(state),
                "cache": None,
            }
            if self._cacheable(state):
                payload = {k: final[k] for k in ("answer", "routing", "citations", "sql", "tool", "verification", "trust")}
                await self.cache.store(state["standalone_query"], tenant, groups, payload, sorted({e["doc_id"] for e in evidence}))

        final["conversation_id"] = conversation_id
        final["request_id"] = state.get("request_id")
        await self.memory.append(tenant, conversation_id, "user", state["query"])
        await self.memory.append(tenant, conversation_id, "assistant", final["answer"])
        await self._observe(state, final)
        return {"final": final, "trace": _trace("finalize", start, cached=bool(cache_hit))}

    # -------------------------------------------------------------- helpers

    async def _observe(self, state: EngineState, final: dict[str, Any]) -> None:
        routing = final.get("routing") or {}
        mode = "blocked" if final.get("blocked") else ("cache" if final.get("cache") else routing.get("mode", "none"))
        routes = routing.get("execute") or ([routing["routes"][0]["route"]] if routing.get("routes") else [])
        verification = final.get("verification") or {}
        latency_ms = round((time.perf_counter() - state.get("started_at", time.perf_counter())) * 1000, 1)
        outcome = (
            "blocked" if final.get("blocked") else "escalated" if final.get("escalation")
            else "abstained" if state.get("abstained") else "answered"
        )
        for route in routes or ["none"]:
            CHAT_REQUESTS.labels(mode, route, outcome).inc()
        CHAT_LATENCY.labels(mode).observe(latency_ms / 1000)
        if verification.get("groundedness") is not None:
            GROUNDEDNESS.observe(verification["groundedness"])
        if final.get("trust") is not None:
            TRUST.observe(final["trust"])

        nodes: dict[str, float] = {}
        for entry in state.get("trace", []):
            nodes[entry["node"]] = nodes.get(entry["node"], 0.0) + entry["ms"]
        try:
            await self.insights.record(state["tenant_id"], {
                "request_id": state.get("request_id"),
                "ts": datetime.now(timezone.utc).isoformat(),
                "query": state["query"],
                "mode": mode,
                "routes": routes,
                "outcome": outcome,
                "cache_hit": bool(final.get("cache")),
                "escalated": bool(final.get("escalation")),
                "abstained": bool(state.get("abstained")),
                "blocked": bool(final.get("blocked")),
                "trust": final.get("trust"),
                "groundedness": verification.get("groundedness"),
                "latency_ms": latency_ms,
                "nodes": nodes,
            })
        except Exception:
            log.warning("Failed to record insights", exc_info=True)

    def _cacheable(self, state: EngineState) -> bool:
        routing = state.get("routing") or {}
        verification = state.get("verification") or {}
        return (
            routing.get("mode") in {"single", "parallel", "plan"}
            and set(routing.get("execute", [])) <= {"A", "B"}
            and bool(state.get("evidence"))
            and verification.get("passed", False)
            and not state.get("abstained")
            and not state.get("is_followup")
            and not state.get("escalation")
        )

    @staticmethod
    def _trust(state: EngineState) -> float | None:
        routing = state.get("routing")
        if not routing or state.get("escalation") or state.get("clarification"):
            return None
        components: list[float] = []
        verification = state.get("verification") or {}
        evidence = state.get("evidence") or []
        if state.get("abstained"):
            return round(0.25 * routing["confidence"], 3)
        if verification.get("groundedness") is not None:
            components.append(verification["groundedness"])
        if evidence:
            components.append(min(1.0, evidence[0]["score"] / 0.5))
        sql_result = state.get("sql_result")
        if sql_result:
            components.append(0.0 if sql_result.get("error") else (verification.get("numeric") or 1.0))
        tool_result = state.get("tool_result")
        if tool_result:
            components.append(1.0 if tool_result["status"] in {"ok", "needs_confirmation"} else 0.2)
        base = sum(components) / len(components) if components else 0.4
        return round(0.75 * base + 0.25 * routing["confidence"], 3)
