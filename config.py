"""Settings for the hotels MCP server and agent. All values come from env / .env."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore" so leftover keys from other modules' .env don't blow up
    # startup (we share one .env with the rest of tripon-mcp-service).
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    yarvel_url: str = Field(default="https://dev-hasura.tripon.io/v1/graphql", alias="YARVEL_URL")
    yarvel_secret: str | None = Field(default=None, alias="YARVEL_SECRET")
    yarvel_org_id: str | None = Field(default=None, alias="YARVEL_ORG_ID")   # UUID string, never int
    yarvel_agency_id: str | None = Field(default=None, alias="YARVEL_AGENCY_ID")
    yarvel_username: str | None = Field(default=None, alias="YARVEL_USERNAME")
    yarvel_password: str | None = Field(default=None, alias="YARVEL_PASSWORD")

    # loginRihla (and some mutations) reject requests without a sender-ip header.
    # Get yours with `curl ifconfig.me`. Only needed on the JWT/login path.
    sender_ip: str | None = Field(default=None, alias="TRIPON_SENDER_IP")

    auth_mode: Literal["admin_secret", "forward_jwt"] = Field(default="admin_secret", alias="HOTELS_AUTH_MODE")
    refresh_window_minutes: int = Field(default=30, alias="HOTELS_REFRESH_WINDOW_MINUTES")
    request_timeout: float = Field(default=30.0, alias="HOTELS_REQUEST_TIMEOUT")
    transport: str = Field(default="stdio", alias="HOTELS_TRANSPORT")

    # Fallbacks when the caller omits them (the search API rejects nulls).
    # USD + US are neutral, non-market-specific defaults; pass the customer's
    # real currency/nationality per request when known, since nationality
    # changes supplier pricing.
    default_currency: str = Field(default="USD", alias="HOTELS_DEFAULT_CURRENCY")
    default_nationality: str = Field(default="US", alias="HOTELS_DEFAULT_NATIONALITY")

    # search() is async: it returns a uuid, then getSearchResults fills in as
    # suppliers answer. These bound the poll in search_hotel_availability.
    availability_max_polls: int = Field(default=6, alias="HOTELS_AVAILABILITY_MAX_POLLS")
    availability_poll_seconds: float = Field(default=1.0, alias="HOTELS_AVAILABILITY_POLL_SECONDS")

    # LLM (OpenRouter, OpenAI-compatible chat completions).
    # Durable memory (Layer 2). "local" needs nothing; "graphiti" needs FalkorDB
    # plus an embeddings provider — OpenRouter serves chat completions only.
    memory_backend: Literal["local", "graphiti"] = Field(default="local", alias="MEMORY_BACKEND")
    memory_top_k: int = Field(default=8, alias="MEMORY_TOP_K")
    falkordb_host: str = Field(default="localhost", alias="FALKORDB_HOST")
    falkordb_port: int = Field(default=6379, alias="FALKORDB_PORT")
    graphiti_llm_model: str = Field(default="anthropic/claude-haiku-4.5", alias="GRAPHITI_LLM_MODEL")
    # "local" needs no key and no download; "openai" needs a real embeddings endpoint.
    graphiti_embedder: Literal["local", "openai"] = Field(default="local", alias="GRAPHITI_EMBEDDER")
    graphiti_embedding_dim: int = Field(default=1024, alias="GRAPHITI_EMBEDDING_DIM")
    graphiti_embedder_api_key: str | None = Field(default=None, alias="GRAPHITI_EMBEDDER_API_KEY")
    graphiti_embedder_base_url: str | None = Field(default=None, alias="GRAPHITI_EMBEDDER_BASE_URL")
    graphiti_embedder_model: str = Field(default="text-embedding-3-small", alias="GRAPHITI_EMBEDDER_MODEL")

    # Web enrichment. "openrouter" reuses the key above — no second provider.
    web_search_backend: Literal["none", "openrouter"] = Field(default="none", alias="WEB_SEARCH_BACKEND")
    web_search_model: str | None = Field(default=None, alias="WEB_SEARCH_MODEL")
    enrichment_index_path: str = Field(default="enrichment_index.sqlite3",
                                       alias="ENRICHMENT_INDEX_PATH")
    web_openmeteo_enabled: bool = Field(default=True, alias="WEB_OPENMETEO_ENABLED")
    web_playwright_enabled: bool = Field(default=False, alias="WEB_PLAYWRIGHT_ENABLED")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")
    web_search_cache_seconds: int = Field(default=3600, alias="WEB_SEARCH_CACHE_SECONDS")

    # Alternates the chat page can switch to. Each carries its own key, because
    # they are separate OpenRouter accounts rather than one account's model list.
    openrouter_model_b: str | None = Field(default=None, alias="OPENROUTER_MODEL_B")
    openrouter_api_key_b: str | None = Field(default=None, alias="OPENROUTER_API_KEY_B")
    openrouter_model_c: str | None = Field(default=None, alias="OPENROUTER_MODEL_C")
    openrouter_api_key_c: str | None = Field(default=None, alias="OPENROUTER_API_KEY_C")
    openrouter_model_d: str | None = Field(default=None, alias="OPENROUTER_MODEL_D")
    openrouter_api_key_d: str | None = Field(default=None, alias="OPENROUTER_API_KEY_D")
    openrouter_model_e: str | None = Field(default=None, alias="OPENROUTER_MODEL_E")
    openrouter_api_key_e: str | None = Field(default=None, alias="OPENROUTER_API_KEY_E")
    openrouter_model_f: str | None = Field(default=None, alias="OPENROUTER_MODEL_F")
    openrouter_api_key_f: str | None = Field(default=None, alias="OPENROUTER_API_KEY_F")
    openrouter_model_g: str | None = Field(default=None, alias="OPENROUTER_MODEL_G")
    openrouter_api_key_g: str | None = Field(default=None, alias="OPENROUTER_API_KEY_G")
    openrouter_model_h: str | None = Field(default=None, alias="OPENROUTER_MODEL_H")
    openrouter_api_key_h: str | None = Field(default=None, alias="OPENROUTER_API_KEY_H")
    openrouter_model_i: str | None = Field(default=None, alias="OPENROUTER_MODEL_I")
    openrouter_api_key_i: str | None = Field(default=None, alias="OPENROUTER_API_KEY_I")
    openrouter_model_j: str | None = Field(default=None, alias="OPENROUTER_MODEL_J")
    openrouter_api_key_j: str | None = Field(default=None, alias="OPENROUTER_API_KEY_J")

    # A slot may live on a different OpenAI-compatible host — a Groq key (gsk_…)
    # gets "Missing Authentication header" from openrouter.ai, because it is not
    # an OpenRouter key at all. Unset means the slot uses OPENROUTER_BASE_URL.
    openrouter_base_url_b: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_B")
    openrouter_base_url_c: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_C")
    openrouter_base_url_d: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_D")
    openrouter_base_url_e: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_E")
    openrouter_base_url_f: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_F")
    openrouter_base_url_g: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_G")
    openrouter_base_url_h: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_H")
    openrouter_base_url_i: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_I")
    openrouter_base_url_j: str | None = Field(default=None, alias="OPENROUTER_BASE_URL_J")

    llm_provider: Literal["openrouter"] = Field(default="openrouter", alias="LLM_PROVIDER")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_model: str = Field(default="anthropic/claude-3-haiku", alias="OPENROUTER_MODEL")
    openrouter_max_tokens: int = Field(default=2000, alias="OPENROUTER_MAX_TOKENS")
    agent_max_iterations: int = Field(default=8, alias="HOTELS_AGENT_MAX_ITERATIONS")
    # One JSON line per completed turn, appended. Unset writes no file — the
    # record still goes to the logger either way. Set it when you need to answer
    # "which model, which tools, did verify pass" about a turn that has scrolled
    # away, which is every question worth asking after a demo.
    run_log_path: str | None = Field(default=None, alias="HOTELS_RUN_LOG")

    @property
    def yarvel_available(self) -> bool:
        return bool(self.yarvel_url and self.yarvel_secret and self.yarvel_org_id)


    def model_options(self) -> list[dict[str, str]]:
        """The models the chat page offers, first one being the default. Each
        carries the host it is reached on, which is not the same for every slot."""
        slots = [(self.openrouter_model, self.openrouter_api_key, None),
                 (self.openrouter_model_b, self.openrouter_api_key_b, self.openrouter_base_url_b),
                 (self.openrouter_model_c, self.openrouter_api_key_c, self.openrouter_base_url_c),
                 (self.openrouter_model_d, self.openrouter_api_key_d, self.openrouter_base_url_d),
                 (self.openrouter_model_e, self.openrouter_api_key_e, self.openrouter_base_url_e),
                 (self.openrouter_model_f, self.openrouter_api_key_f, self.openrouter_base_url_f),
                 (self.openrouter_model_g, self.openrouter_api_key_g, self.openrouter_base_url_g),
                 (self.openrouter_model_h, self.openrouter_api_key_h, self.openrouter_base_url_h),
                 (self.openrouter_model_i, self.openrouter_api_key_i, self.openrouter_base_url_i),
                 (self.openrouter_model_j, self.openrouter_api_key_j, self.openrouter_base_url_j)]
        return [{"model": m, "api_key": k, "base_url": b or self.openrouter_base_url}
                for m, k, b in slots if m and k]

    def credentials_for(self, model: str | None) -> tuple[str, str | None]:
        """Match a requested model to the key that pays for it. An unknown name
        falls back to the default rather than borrowing another account's key."""
        options = self.model_options()
        if model:
            for option in options:
                if option["model"] == model:
                    return option["model"], option["api_key"]
        return (self.openrouter_model, self.openrouter_api_key)

    def base_url_for(self, model: str | None) -> str:
        """The host that answers for this model. Kept separate from
        credentials_for so its two-value return stays as callers expect."""
        if model:
            for option in self.model_options():
                if option["model"] == model:
                    return option["base_url"]
        return self.openrouter_base_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
