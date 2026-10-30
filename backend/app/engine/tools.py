import ast
import json
import logging
import operator
import re
import secrets
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from redis.asyncio import Redis

from app.engine.router import CONFIRM
from app.llm.client import LLMClient

log = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression using + - * / ** % and parentheses.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the current date and time in an IANA timezone.",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "UTC"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create an IT support ticket. This has side effects and requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                },
                "required": ["title"],
            },
        },
    },
]
SIDE_EFFECT_TOOLS = frozenset({"create_support_ticket"})

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_EXPRESSION = re.compile(r"[-+*/^%().\d\s]*\d[-+*/^%().\d\s]*")
_WORD_OPERATORS = {" plus ": " + ", " minus ": " - ", " times ": " * ", " multiplied by ": " * ", " divided by ": " / ", " x ": " * "}


def safe_eval(expression: str) -> float:
    if len(expression) > 200:
        raise ValueError("Expression is too long.")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("Exponent is too large.")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("Unsupported expression.")

    return evaluate(tree)


def extract_expression(text: str) -> str | None:
    lowered = f" {text.lower()} "
    for word, symbol in _WORD_OPERATORS.items():
        lowered = lowered.replace(word, symbol)
    candidates = [m.group(0).strip() for m in _EXPRESSION.finditer(lowered)]
    candidates = [c for c in candidates if re.search(r"\d\s*[-+*/^%]\s*[\d(]", c)]
    return max(candidates, key=len) if candidates else None


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else f"{value:,.6g}"


def heuristic_tool_call(query: str) -> dict[str, Any] | None:
    expression = extract_expression(query)
    if expression:
        return {"name": "calculator", "arguments": {"expression": expression}}
    if re.search(r"\b(time|date|day|today)\b", query, re.IGNORECASE):
        return {"name": "current_datetime", "arguments": {"timezone": "UTC"}}
    if re.search(r"\bticket\b", query, re.IGNORECASE):
        title = re.sub(
            r"^.*?\bticket\b\s*(for|about|regarding|:)?\s*", "", query, count=1, flags=re.IGNORECASE
        ).strip(" .") or query.strip()
        priority_match = re.search(r"\bP([1-4])\b", query, re.IGNORECASE)
        if priority_match:
            priority = f"P{priority_match.group(1)}"
        elif re.search(r"\b(urgent|critical|outage|down)\b", query, re.IGNORECASE):
            priority = "P1"
        elif re.search(r"\bhigh\b", query, re.IGNORECASE):
            priority = "P2"
        elif re.search(r"\blow\b", query, re.IGNORECASE):
            priority = "P4"
        else:
            priority = "P3"
        return {"name": "create_support_ticket", "arguments": {"title": title[:200], "priority": priority}}
    return None


class ToolAgent:
    def __init__(self, llm: LLMClient, redis: Redis) -> None:
        self._llm = llm
        self._redis = redis

    @staticmethod
    def pending_key(tenant: str, conversation_id: str) -> str:
        return f"pending:{tenant}:{conversation_id}"

    async def run(self, query: str, tenant: str, conversation_id: str) -> dict[str, Any]:
        key = self.pending_key(tenant, conversation_id)
        if CONFIRM.match(query):
            raw = await self._redis.get(key)
            if not raw:
                return self._result(None, {}, "error", "There is no pending action to confirm.")
            await self._redis.delete(key)
            action = json.loads(raw)
            return await self._execute(action["name"], action["arguments"], tenant)

        call = await self._select(query)
        if call is None:
            return self._result(None, {}, "error", "I couldn't find a tool that can handle this request.")
        if call["name"] in SIDE_EFFECT_TOOLS:
            args = call["arguments"]
            args.setdefault("priority", "P3")
            await self._redis.set(key, json.dumps(call), ex=600)
            return self._result(
                call["name"], args, "needs_confirmation",
                f"I'm ready to create a **{args['priority']}** support ticket titled \"{args.get('title', '')}\". "
                "Reply **confirm** to proceed.",
            )
        return await self._execute(call["name"], call["arguments"], tenant)

    async def _select(self, query: str) -> dict[str, Any] | None:
        if not self._llm.is_mock:
            try:
                call = await self._llm.complete_with_tools([{"role": "user", "content": query}], TOOL_SCHEMAS)
                if call and call["name"] in {t["function"]["name"] for t in TOOL_SCHEMAS}:
                    return call
            except Exception as exc:
                log.warning("LLM tool selection failed: %s", exc.__class__.__name__)
        return heuristic_tool_call(query)

    async def _execute(self, name: str, args: dict[str, Any], tenant: str) -> dict[str, Any]:
        try:
            if name == "calculator":
                expression = str(args.get("expression", ""))
                value = safe_eval(expression)
                return self._result(name, args, "ok", f"{expression.strip()} = **{_format_number(value)}**")
            if name == "current_datetime":
                tz_name = str(args.get("timezone") or "UTC")
                try:
                    tz = timezone.utc if tz_name.upper() == "UTC" else ZoneInfo(tz_name)
                except (ZoneInfoNotFoundError, ValueError):
                    tz, tz_name = timezone.utc, "UTC"
                now = datetime.now(tz)
                return self._result(name, args, "ok", f"The current date and time is **{now:%A, %d %B %Y %H:%M}** ({tz_name}).")
            if name == "create_support_ticket":
                ticket = {
                    "id": f"INC-{secrets.token_hex(3).upper()}",
                    "title": str(args.get("title", ""))[:200],
                    "priority": args.get("priority", "P3"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                await self._redis.lpush(f"tickets:{tenant}", json.dumps(ticket))
                await self._redis.ltrim(f"tickets:{tenant}", 0, 199)
                return self._result(
                    name, args, "ok", f"Created support ticket **{ticket['id']}** ({ticket['priority']}): {ticket['title']}"
                )
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return self._result(name, args, "error", f"The {name} tool could not complete: {exc}")
        return self._result(name, args, "error", f"Unknown tool {name}.")

    @staticmethod
    def _result(tool: str | None, args: dict[str, Any], status: str, output: str) -> dict[str, Any]:
        return {"tool": tool, "args": args, "status": status, "output": output, "side_effect": tool in SIDE_EFFECT_TOOLS}
