from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Root .env first, then a local override when running from backend/.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AdvanceRAG Engine"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]
    log_level: str = "INFO"

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql://advance_rag@localhost:5432/advance_rag"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    embedding_dim: int = 1024

    router_single_threshold: float = 0.85
    router_parallel_threshold: float = 0.60
    groundedness_threshold: float = 0.80
    semantic_cache_threshold: float = 0.95
    cache_ttl_seconds: int = 86400

    retrieval_top_k: int = 30
    context_token_budget: int = 3000
    chunk_max_tokens: int = 350
    chunk_overlap_sentences: int = 1

    max_upload_mb: int = 20
    seed_sample_docs: bool = True

    # Authentication: "header" trusts X-Tenant-Id/X-User-Id/X-User-Groups (local demo only);
    # "oidc" validates bearer tokens. Claim defaults match Microsoft Entra ID.
    auth_mode: Literal["header", "oidc"] = "header"
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_algorithms: list[str] = ["RS256"]
    oidc_tenant_claim: str = "tid"
    oidc_user_claim: str = "oid"
    oidc_groups_claim: str = "roles"
    oidc_default_tenant: str | None = None

    # RAG lab / learning playground
    lab_enabled: bool = True
    lab_max_sessions: int = 24
    lab_max_sessions_per_user: int = 3
    lab_session_ttl_seconds: int = 3600
    lab_max_corpus_chars: int = 2_000_000

    # Scalability
    db_pool_min: int = 1
    db_pool_max: int = 10
    max_inflight_requests: int = 100
    rate_limit_capacity: int = 30
    rate_limit_refill_per_sec: float = 0.5
    run_embedded_worker: bool = True
    worker_concurrency: int = 4
    worker_metrics_port: int = 9100

    # Observability
    json_logs: bool = True
    otel_exporter_otlp_endpoint: str | None = None
    applicationinsights_connection_string: str | None = None

    @property
    def use_mock_llm(self) -> bool:
        return not self.azure_openai_endpoint

    @property
    def embedding_model_id(self) -> str:
        return "mock-hash" if self.use_mock_llm else self.azure_openai_embedding_deployment


@lru_cache
def get_settings() -> Settings:
    return Settings()
