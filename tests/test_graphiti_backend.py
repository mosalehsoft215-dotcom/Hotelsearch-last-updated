"""The Graphiti backend's own logic, without the package or a database.

Every bug that reached production lived here: the group_id charset, the empty
query, and the per-group graph. graphiti_core is stubbed so these run in CI;
tests/test_live.py exercises the same paths against a real container.
"""
import re
import sys
import types
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace

import pytest

from memory import (
    Episode, EpisodeKind, GraphitiGraphBackend, GraphitiMemory, graphiti_group_id,
    org_group, user_group,
)

USER = "alice@corp.com"
ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"
T1 = datetime(2026, 3, 2, tzinfo=timezone.utc)
GRAPHITI_CHARSET = re.compile(r"^[A-Za-z0-9_-]+$")


def edge(fact, group, valid_at=T1, invalid_at=None):
    return SimpleNamespace(fact=fact, group_id=group, valid_at=valid_at, invalid_at=invalid_at)


class FakeDriver:
    def __init__(self, database=None):
        self.database = database

    def clone(self, database):
        return FakeDriver(database)


class FakeClient:
    def __init__(self, search_edges=None):
        self.driver = FakeDriver()
        self.episodes: list[dict] = []
        self.searches: list[dict] = []
        self._search_edges = search_edges or []

    async def add_episode(self, **kwargs):
        self.episodes.append(kwargs)

    async def search(self, query, group_ids=None, num_results=None):
        self.searches.append({"query": query, "group_ids": group_ids,
                              "num_results": num_results})
        return self._search_edges


@pytest.fixture
def graphiti(monkeypatch):
    """Stub the pieces of graphiti_core the backend imports at call time."""

    class EpisodeType(Enum):
        message = "message"
        text = "text"
        json = "json"

    class GroupsEdgesNotFoundError(Exception):
        pass

    class EntityEdge:
        by_group: dict[str, list] = {}
        calls: list[tuple] = []

        @classmethod
        async def get_by_group_ids(cls, driver, group_ids, limit=None):
            cls.calls.append((driver.database, tuple(group_ids), limit))
            found = [e for g in group_ids for e in cls.by_group.get(g, [])]
            if not found:
                raise GroupsEdgesNotFoundError(group_ids)
            return found

    EntityEdge.by_group = {}
    EntityEdge.calls = []

    for name, attrs in (("graphiti_core", {}),
                        ("graphiti_core.nodes", {"EpisodeType": EpisodeType}),
                        ("graphiti_core.edges", {"EntityEdge": EntityEdge}),
                        ("graphiti_core.errors", {"GroupsEdgesNotFoundError": GroupsEdgesNotFoundError})):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(EpisodeType=EpisodeType, EntityEdge=EntityEdge,
                           NotFound=GroupsEdgesNotFoundError)


# ---- group id: Graphiti allows [A-Za-z0-9_-] only ----

def test_group_id_is_sanitised_for_graphiti():
    assert graphiti_group_id(user_group(USER)) == "user_alice_corp_com"
    assert graphiti_group_id(org_group(ORG)) == "org_9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def test_sanitised_group_id_matches_the_charset_rule():
    for group in (user_group(USER), org_group(ORG), user_group("ops_triage_agent")):
        assert GRAPHITI_CHARSET.match(graphiti_group_id(group))


def test_sanitising_twice_gives_the_same_group():
    once = graphiti_group_id(user_group(USER))
    assert graphiti_group_id(once) == once      # a read lands where the write went


# ---- writes ----

@pytest.mark.asyncio
async def test_add_episode_sends_the_required_arguments(graphiti):
    client = FakeClient()
    backend = GraphitiGraphBackend(client)
    await backend.add_episode(Episode(
        name="preference", body="I book 5-star only.", kind=EpisodeKind.message,
        group_id=user_group(USER), source_description="user statement",
        reference_time=T1, key="hotel_stars"))

    sent = client.episodes[0]
    assert sent["group_id"] == "user_alice_corp_com"
    assert sent["source_description"] == "user statement"   # required by the API
    assert sent["reference_time"] == T1                     # required by the API
    assert sent["source"] is graphiti.EpisodeType.message
    assert sent["episode_body"] == "I book 5-star only."


@pytest.mark.asyncio
async def test_episode_kind_maps_to_the_graphiti_enum(graphiti):
    client = FakeClient()
    backend = GraphitiGraphBackend(client)
    for kind in (EpisodeKind.message, EpisodeKind.text, EpisodeKind.json):
        await backend.add_episode(Episode(
            name="e", body="b", kind=kind, group_id=user_group(USER),
            source_description="s", reference_time=T1))
    assert [e["source"].value for e in client.episodes] == ["message", "text", "json"]


# ---- reads ----

@pytest.mark.asyncio
async def test_search_with_a_query_uses_hybrid_search(graphiti):
    client = FakeClient([edge("I book 5-star only.", "user_alice_corp_com")])
    facts = await GraphitiGraphBackend(client).search(
        "hotel standard", [user_group(USER)], limit=5)

    assert [f.statement for f in facts] == ["I book 5-star only."]
    call = client.searches[0]
    assert call["group_ids"] == ["user_alice_corp_com"]     # sanitised, and a list
    assert call["num_results"] == 5
    assert graphiti.EntityEdge.calls == []                  # no listing fallback


@pytest.mark.asyncio
async def test_empty_query_lists_the_group_instead_of_searching(graphiti):
    graphiti.EntityEdge.by_group = {
        "user_alice_corp_com": [edge("I book 5-star only.", "user_alice_corp_com")]}
    client = FakeClient()
    facts = await GraphitiGraphBackend(client).search("  ", [user_group(USER)], limit=10)

    assert [f.statement for f in facts] == ["I book 5-star only."]
    assert client.searches == []                            # hybrid search skipped
    database, groups, limit = graphiti.EntityEdge.calls[0]
    assert database == "user_alice_corp_com"                # driver cloned onto the group
    assert groups == ("user_alice_corp_com",) and limit == 10


@pytest.mark.asyncio
async def test_listing_skips_groups_with_nothing_recorded(graphiti):
    graphiti.EntityEdge.by_group = {"org_" + ORG.replace("-", "-"): []}
    client = FakeClient()
    facts = await GraphitiGraphBackend(client).search(
        "", [org_group(ORG), user_group(USER)], limit=10)
    assert facts == []                                      # not found is tolerated
    assert len(graphiti.EntityEdge.calls) == 2              # both groups attempted


@pytest.mark.asyncio
async def test_superseded_facts_are_hidden_unless_asked_for(graphiti):
    edges = [edge("I book 4-star.", "user_alice_corp_com", invalid_at=T1),
             edge("I book 5-star only.", "user_alice_corp_com")]
    client = FakeClient(edges)
    backend = GraphitiGraphBackend(client)

    current = await backend.search("hotels", [user_group(USER)])
    assert [f.statement for f in current] == ["I book 5-star only."]

    everything = await backend.search("hotels", [user_group(USER)], include_invalid=True)
    assert len(everything) == 2
    assert [f.current for f in everything] == [False, True]


# ---- through the wrapper the rest of the service uses ----

@pytest.mark.asyncio
async def test_memory_writes_and_reads_over_the_graphiti_backend(graphiti):
    client = FakeClient([edge("I book 5-star only.", "user_alice_corp_com")])
    memory = GraphitiMemory(GraphitiGraphBackend(client))

    await memory.add_user_episode("I book 5-star only.", USER, key="hotel_stars",
                                  reference_time=T1)
    assert client.episodes[0]["group_id"] == "user_alice_corp_com"

    context = await memory.get_context("hotel standard", username=USER)
    assert context and "5-star" in context
    assert client.searches[-1]["group_ids"] == ["user_alice_corp_com"]
