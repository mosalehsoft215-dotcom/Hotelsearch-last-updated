"""Durable memory for the agents — Layer 2 of the memory architecture.

SessionMemory in runtime.py is Layer 1: per session, lost when the process ends.
This module is what survives: facts about a user or an organization, on a
temporal knowledge graph, with the old value kept when a fact changes.

Two backends behind one interface:

  local     an in-process graph. No infrastructure, deterministic, used for the
            demo and the tests.
  graphiti  graphiti_core against FalkorDB. Extraction and embeddings go to the
            configured providers.

Capture stays explicit. The typed add_* methods below are the only way in —
there is deliberately no raw add_episode() pass-through, so supplier payloads
and tool results can never leak into memory on their own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


class EpisodeKind(str, Enum):
    message = "message"   # something the user said
    text = "text"         # a summary an agent wrote
    json = "json"         # a structured payload, e.g. org defaults


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Episode:
    """Raw input, stored forever and never edited."""
    name: str
    body: str
    kind: EpisodeKind
    group_id: str
    source_description: str
    reference_time: datetime
    key: str | None = None


@dataclass
class Fact:
    """A statement with the two event-time bounds. `invalid_at` set means the
    statement used to be true — the row stays, it is not removed."""
    statement: str
    group_id: str
    key: str | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)

    @property
    def current(self) -> bool:
        return self.invalid_at is None


class MemoryBackend(Protocol):
    async def add_episode(self, episode: Episode) -> None: ...

    async def search(self, query: str, group_ids: list[str], limit: int = 10,
                     include_invalid: bool = False) -> list[Fact]: ...


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _stems(text: str) -> set[str]:
    # crude stemming so "hotel" matches "hotels"; the real backend uses
    # embeddings plus BM25 plus a graph walk instead.
    return {t[:4] for t in _tokens(text) if len(t) > 2}


def _relevance(query: str, statement: str) -> float:
    wanted = _stems(query)
    if not wanted:
        return 0.0
    return len(wanted & _stems(statement)) / len(wanted)


class LocalGraphBackend:
    """In-process store with the same temporal behaviour as the graph.

    A fact carries an optional `key` (for example "cabin_class"). When a new
    fact arrives with a key that already exists in the same group, the previous
    one is closed off with `invalid_at` instead of being overwritten. The real
    Graphiti backend reaches the same result through LLM entity resolution; here
    the key makes it deterministic, which is what a demo and a test need.
    """

    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.facts: list[Fact] = []

    async def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if episode.key:
            for fact in self.facts:
                if (fact.group_id == episode.group_id and fact.key == episode.key
                        and fact.invalid_at is None):
                    fact.invalid_at = episode.reference_time
        self.facts.append(Fact(statement=episode.body, group_id=episode.group_id,
                               key=episode.key, valid_at=episode.reference_time))

    async def search(self, query: str, group_ids: list[str], limit: int = 10,
                     include_invalid: bool = False) -> list[Fact]:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        scored: list[tuple[float, datetime, Fact]] = []
        for fact in self.facts:
            if fact.group_id not in group_ids:
                continue
            if fact.invalid_at is not None and not include_invalid:
                continue
            # relevance first, then recency — a fact that does not match the
            # wording is still worth carrying if the group has room in top-k.
            scored.append((_relevance(query, fact.statement),
                           fact.valid_at or epoch, fact))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [fact for _, _, fact in scored[:limit]]


_GRAPHITI_GROUP_ILLEGAL = re.compile(r"[^A-Za-z0-9_-]")


def graphiti_group_id(group: str) -> str:
    """Our group ids read as "user:alice@corp.com" / "org:<uuid>", but Graphiti
    rejects anything outside [A-Za-z0-9_-] with GroupIdValidationError. Map the
    offending characters to underscores, and only at this boundary — the local
    backend, the tests and the console all keep the readable form.

    The mapping is deterministic, so a write and a later read land on the same
    group; it is not injective ("a.b" and "a_b" collide), which is fine for
    usernames and UUIDs.
    """
    return _GRAPHITI_GROUP_ILLEGAL.sub("_", group)


class GraphitiGraphBackend:
    """graphiti_core against FalkorDB.

    Note the real signatures: add_episode needs source_description and
    reference_time, search takes group_ids as a list, and search returns
    EntityEdge objects — the context string is assembled by GraphitiMemory.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Any) -> "GraphitiGraphBackend":
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient

        from graphiti_embedder import build_embedder

        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for Graphiti extraction")
        # extraction and reranking go to OpenRouter; vectors come from the
        # embedder chosen in config (local by default — OpenRouter has no
        # embeddings endpoint).
        llm_config = LLMConfig(api_key=settings.openrouter_api_key,
                               base_url=settings.openrouter_base_url,
                               model=settings.graphiti_llm_model)
        embedder = build_embedder(settings)
        client = Graphiti(
            graph_driver=FalkorDriver(host=settings.falkordb_host, port=settings.falkordb_port),
            llm_client=OpenAIClient(config=llm_config),
            embedder=embedder,
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )
        return cls(client)

    async def build_indices(self) -> None:
        await self._client.build_indices_and_constraints()

    async def add_episode(self, episode: Episode) -> None:
        from graphiti_core.nodes import EpisodeType
        await self._client.add_episode(
            name=episode.name,
            episode_body=episode.body,
            source_description=episode.source_description,
            reference_time=episode.reference_time,
            source=EpisodeType[episode.kind.value],
            group_id=graphiti_group_id(episode.group_id),
        )

    async def search(self, query: str, group_ids: list[str], limit: int = 10,
                     include_invalid: bool = False) -> list[Fact]:
        groups = [graphiti_group_id(g) for g in group_ids]
        if query.strip():
            edges = await self._client.search(query, group_ids=groups, num_results=limit)
        else:
            # Hybrid search on an empty string matches nothing here — no vector to
            # compare and no BM25 terms — so "show me everything" (the console's
            # Memory panel, and history()) came back empty while the graph held
            # facts. Listing the group directly is the honest answer to no query.
            edges = await self._all_edges(groups, limit)
        facts = [Fact(statement=e.fact, group_id=getattr(e, "group_id", "") or "",
                      valid_at=getattr(e, "valid_at", None),
                      invalid_at=getattr(e, "invalid_at", None)) for e in edges]
        return facts if include_invalid else [f for f in facts if f.current]

    async def _all_edges(self, groups: list[str], limit: int) -> list[Any]:
        """Every fact edge in these groups, no query.

        FalkorDB gives each group its own graph, so the driver has to be cloned
        onto that database per group — asking the default one returns nothing
        even when the data is there. Neo4j would not need the clone, but it is
        harmless there, so there is no branch on provider.
        """
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        edges: list[Any] = []
        for group in groups:
            driver = self._client.driver
            try:
                driver = driver.clone(group)
            except Exception:      # a driver that does not scope per database
                pass
            try:
                edges.extend(await EntityEdge.get_by_group_ids(
                    driver, group_ids=[group], limit=limit))
            except GroupsEdgesNotFoundError:
                continue           # nothing recorded for this group yet
        return edges[:limit]


def user_group(username: str) -> str:
    return f"user:{username}"


def org_group(org_id: str) -> str:
    return f"org:{org_id}"


class GraphitiMemory:
    """The only integration point the rest of the service uses."""

    def __init__(self, backend: MemoryBackend, top_k: int = 8) -> None:
        self._backend = backend
        self.top_k = top_k

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    async def add_user_episode(self, statement: str, username: str, *,
                               key: str | None = None,
                               kind: EpisodeKind = EpisodeKind.message,
                               reference_time: datetime | None = None) -> None:
        """A preference the user stated, in their own words."""
        await self._backend.add_episode(Episode(
            name="preference", body=statement, kind=kind, group_id=user_group(username),
            source_description="user statement", reference_time=reference_time or _now(),
            key=key))

    async def add_org_episode(self, payload: str | dict, org_id: str, *,
                              key: str | None = None,
                              kind: EpisodeKind = EpisodeKind.json,
                              reference_time: datetime | None = None) -> None:
        """An organization-level default."""
        body = payload if isinstance(payload, str) else str(payload)
        await self._backend.add_episode(Episode(
            name="org_default", body=body, kind=kind, group_id=org_group(org_id),
            source_description="org admin", reference_time=reference_time or _now(),
            key=key))

    async def add_booking_episode(self, summary: str, username: str, *,
                                  key: str | None = None,
                                  reference_time: datetime | None = None) -> None:
        """A short summary written after a booking completes."""
        await self._backend.add_episode(Episode(
            name="booking", body=summary, kind=EpisodeKind.text,
            group_id=user_group(username), source_description="booking completion",
            reference_time=reference_time or _now(), key=key))

    async def add_ops_episode(self, signal: str, agent_name: str, *,
                              key: str | None = None,
                              reference_time: datetime | None = None) -> None:
        """A dead-letter-queue signature seen by the ops triage agent."""
        await self._backend.add_episode(Episode(
            name="ops_pattern", body=signal, kind=EpisodeKind.text,
            group_id=user_group(agent_name), source_description="ops triage",
            reference_time=reference_time or _now(), key=key))

    async def get_context(self, query: str, username: str | None = None,
                          org_id: str | None = None, limit: int | None = None) -> str | None:
        """Facts worth putting in the system prompt. Org defaults first, then the
        user's own — the user's value wins when both describe the same key."""
        if not username and not org_id:
            return None
        limit = limit or self.top_k
        merged: dict[str, Fact] = {}
        order: list[Fact] = []

        async def collect(group: str) -> None:
            for fact in await self._backend.search(query, [group], limit=limit):
                marker = fact.key or fact.statement
                if marker in merged:
                    order[order.index(merged[marker])] = fact   # later scope overrides
                else:
                    order.append(fact)
                merged[marker] = fact

        if org_id:
            await collect(org_group(org_id))
        if username:
            await collect(user_group(username))
        if not order:
            return None
        lines = []
        for fact in order[:limit]:
            since = f" (since {fact.valid_at:%Y-%m-%d})" if fact.valid_at else ""
            lines.append(f"- {fact.statement}{since}")
        return "What we know about this user and organization:\n" + "\n".join(lines)

    async def is_known_ops_pattern(self, signature: str, agent_name: str) -> bool:
        """True when this signature was already recorded and is still open, so the
        agent reports it as known rather than new."""
        for fact in await self._backend.search(signature, [user_group(agent_name)], limit=self.top_k):
            if fact.statement.strip().lower() == signature.strip().lower() and fact.current:
                return True
        return False

    async def history(self, query: str, username: str | None = None,
                      org_id: str | None = None) -> list[Fact]:
        """Everything matching, superseded facts included — nothing is deleted."""
        groups = [g for g in (org_group(org_id) if org_id else None,
                              user_group(username) if username else None) if g]
        return await self._backend.search(query, groups, limit=50, include_invalid=True)


def build_memory(settings: Any) -> GraphitiMemory:
    """local by default; graphiti when MEMORY_BACKEND=graphiti."""
    if settings.memory_backend == "graphiti":
        return GraphitiMemory(GraphitiGraphBackend.from_settings(settings),
                              top_k=settings.memory_top_k)
    return GraphitiMemory(LocalGraphBackend(), top_k=settings.memory_top_k)
