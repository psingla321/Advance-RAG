import contextvars
import json
import logging
import sys

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

from app.config import Settings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
tracer = trace.get_tracer("advance_rag")

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)

# Tenant is intentionally not a label (unbounded cardinality); per-tenant analytics live in InsightsStore.
CHAT_REQUESTS = Counter("advance_rag_chat_requests_total", "Chat requests by outcome", ["mode", "route", "outcome"])
CHAT_LATENCY = Histogram("advance_rag_chat_latency_seconds", "End-to-end chat latency", ["mode"], buckets=_LATENCY_BUCKETS)
NODE_LATENCY = Histogram("advance_rag_node_latency_seconds", "Latency per engine node", ["node"], buckets=_LATENCY_BUCKETS)
CACHE_LOOKUPS = Counter("advance_rag_cache_lookups_total", "Answer cache lookups", ["result"])
GROUNDEDNESS = Histogram("advance_rag_groundedness_score", "Verifier groundedness", buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 1.0))
TRUST = Histogram("advance_rag_trust_score", "Answer trust score", buckets=(0.2, 0.4, 0.6, 0.75, 0.9, 1.0))
LLM_CALLS = Counter("advance_rag_llm_calls_total", "LLM/embedding calls", ["operation", "model"])
LLM_TOKENS = Counter("advance_rag_llm_tokens_total", "LLM tokens", ["kind", "model"])
INFLIGHT = Gauge("advance_rag_inflight_requests", "Chat requests currently in flight")
RATE_LIMITED = Counter("advance_rag_rate_limited_total", "Requests rejected by rate limiting")
SHED = Counter("advance_rag_load_shed_total", "Requests rejected due to overload")
INGEST_JOBS = Counter("advance_rag_ingestion_jobs_total", "Ingestion jobs by status", ["status"])
INGEST_LATENCY = Histogram("advance_rag_ingestion_seconds", "Ingestion duration per document", buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120))
QUEUE_DEPTH = Gauge("advance_rag_ingestion_queue_depth", "Ingestion jobs waiting or in progress")
FEEDBACK = Counter("advance_rag_feedback_total", "User feedback", ["rating"])


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            payload["trace_id"] = format(context.trace_id, "032x")
            payload["span_id"] = format(context.span_id, "016x")
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    for noisy in ("httpx", "httpcore", "azure.core.pipeline.policies.http_logging_policy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_tracing(settings: Settings, service_name: str) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name, "service.version": "0.2.0"}))
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")))
    if settings.applicationinsights_connection_string:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

        provider.add_span_processor(
            BatchSpanProcessor(AzureMonitorTraceExporter(connection_string=settings.applicationinsights_connection_string))
        )
    trace.set_tracer_provider(provider)


def shutdown_tracing() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
