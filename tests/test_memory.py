from datetime import datetime, timezone

import pytest

import memory_tools
from config import Settings
from memory import (
    EpisodeKind, GraphitiMemory, LocalGraphBackend, build_memory, org_group, user_group,
)
from runtime import AgentContext

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"
USER = "m.saleh"
T1 = datetime(2026, 3, 2, tzinfo=timezone.utc)
T2 = datetime(2026, 7, 20, tzinfo=timezone.utc)


def _memory() -> GraphitiMemory:
    return GraphitiMemory(LocalGraphBackend(), top_k=8)


def test_build_memory_defaults_to_local():
    m = build_memory(Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None))
    assert isinstance(m.backend, LocalGraphBackend)


def test_group_ids_scope_by_user_and_org():
    assert user_group(USER) == f"user:{USER}"
    assert org_group(ORG) == f"org:{ORG}"


@pytest.mark.asyncio
async def test_preference_is_recalled_in_a_new_session():
    m = _memory()
    await m.add_user_episode("I book 4-star hotels or better.", USER, key="hotel_stars",
                             reference_time=T1)
    context = await m.get_context("find a hotel", username=USER, org_id=ORG)
    assert context and "4-star" in context


@pytest.mark.asyncio
async def test_new_value_supersedes_and_the_old_one_is_kept():
    m = _memory()
    await m.add_user_episode("I book 4-star hotels.", USER, key="hotel_stars", reference_time=T1)
    await m.add_user_episode("I book 5-star only now.", USER, key="hotel_stars", reference_time=T2)

    context = await m.get_context("hotel standard", username=USER)
    assert "5-star" in context and "4-star" not in context      # only the current fact is told

    history = await m.history("hotel", username=USER)
    old = [f for f in history if "4-star" in f.statement][0]
    assert old.invalid_at == T2 and not old.current              # kept, closed off, not deleted
    assert any(f.current and "5-star" in f.statement for f in history)


@pytest.mark.asyncio
async def test_user_preference_overrides_org_default_on_the_same_key():
    m = _memory()
    await m.add_org_episode("Agency default board is room only.", ORG, key="board",
                            reference_time=T1)
    await m.add_org_episode("Agency currency is USD.", ORG, key="currency", reference_time=T1)
    await m.add_user_episode("I want breakfast included.", USER, key="board", reference_time=T2)

    context = await m.get_context("board and currency", username=USER, org_id=ORG)
    assert "breakfast" in context           # the user's value won
    assert "room only" not in context       # the org default was replaced, per key
    assert "USD" in context                 # the untouched org default still shows


@pytest.mark.asyncio
async def test_org_only_and_no_identity():
    m = _memory()
    await m.add_org_episode("Agency currency is USD.", ORG, key="currency")
    assert "USD" in (await m.get_context("currency", org_id=ORG) or "")
    assert await m.get_context("anything") is None


@pytest.mark.asyncio
async def test_ops_signature_is_known_on_the_second_session():
    m = _memory()
    sig = "QuotationToBookingV2:validation_error"
    assert await m.is_known_ops_pattern(sig, "ops_triage_agent") is False
    await m.add_ops_episode(sig, "ops_triage_agent", key=sig)
    assert await m.is_known_ops_pattern(sig, "ops_triage_agent") is True
    assert await m.is_known_ops_pattern("BookAndIssue:unknown", "ops_triage_agent") is False


@pytest.mark.asyncio
async def test_episodes_are_kept_verbatim():
    m = _memory()
    await m.add_user_episode("I always ask for a high floor.", USER, key="room")
    await m.add_booking_episode("Booked a 5-star in Jeddah, room only, 3 nights.", USER)
    episodes = m.backend.episodes
    assert [e.kind for e in episodes] == [EpisodeKind.message, EpisodeKind.text]
    assert episodes[0].body == "I always ask for a high floor."
    assert all(e.group_id == user_group(USER) for e in episodes)


# ---- the tools the agents call ----

@pytest.mark.asyncio
async def test_remember_and_recall_tools():
    m = _memory()
    ctx = AgentContext(org_id=ORG, username=USER, memory=m)
    stored = await memory_tools.remember_preference("I prefer breakfast included.",
                                                    key="board", ctx=ctx)
    assert stored["stored"] is True
    recalled = await memory_tools.recall_preferences("board", ctx=ctx)
    assert "breakfast" in recalled["context"]


@pytest.mark.asyncio
async def test_remember_needs_an_identity():
    ctx = AgentContext(org_id=ORG, memory=_memory())      # no username
    assert (await memory_tools.remember_preference("x", ctx=ctx))["stored"] is False


@pytest.mark.asyncio
async def test_record_ops_pattern_tool_reports_known():
    m = _memory()
    ctx = AgentContext(org_id=ORG, memory=m)
    first = await memory_tools.record_ops_pattern("BookAndIssue:unknown", ctx=ctx)
    assert first == {"recorded": True, "known": False, "signature": "BookAndIssue:unknown"}
    again = await memory_tools.record_ops_pattern("BookAndIssue:unknown", ctx=ctx)
    assert again["known"] is True and again["recorded"] is False


@pytest.mark.asyncio
async def test_tools_are_inert_without_memory():
    ctx = AgentContext(org_id=ORG, username=USER)          # memory is None
    assert (await memory_tools.remember_preference("x", ctx=ctx))["stored"] is False
    assert (await memory_tools.recall_preferences("x", ctx=ctx))["context"] is None
    assert (await memory_tools.record_ops_pattern("s", ctx=ctx))["known"] is False


# ---- wiring: retrieval happens once at session start and reaches the prompt ----

@pytest.mark.asyncio
async def test_memory_context_is_injected_into_the_prompt():
    from agents.hotel_search_agent import HotelSearchAgent
    from runtime import LLMResponse

    class OneShot:
        def __init__(self): self.prompts = []
        async def complete(self, messages, tools=None):
            self.prompts.append(messages[0]["content"])
            return LLMResponse(content="ok")

    m = _memory()
    await m.add_user_episode("I book 5-star only.", USER, key="hotel_stars")
    ctx = AgentContext(org_id=ORG, username=USER, memory=m)
    llm = OneShot()
    await HotelSearchAgent().run(ctx, "find me a hotel", llm)
    assert "5-star" in llm.prompts[0]
    assert ctx.memory_context is not None


@pytest.mark.asyncio
async def test_retrieval_is_cached_for_the_session():
    from agents.hotel_search_agent import HotelSearchAgent
    from runtime import LLMResponse

    class Counting:
        def __init__(self, inner): self.inner, self.searches = inner, 0
        async def add_episode(self, episode): await self.inner.add_episode(episode)
        async def search(self, *a, **k):
            self.searches += 1
            return await self.inner.search(*a, **k)

    class Chat:
        async def complete(self, messages, tools=None): return LLMResponse(content="ok")

    backend = Counting(LocalGraphBackend())
    m = GraphitiMemory(backend)
    await m.add_user_episode("I book 5-star only.", USER, key="hotel_stars")
    ctx = AgentContext(org_id=ORG, username=USER, memory=m)
    agent, llm = HotelSearchAgent(), Chat()
    await agent.run(ctx, "first turn", llm)
    after_first = backend.searches
    await agent.run(ctx, "second turn", llm, history=None)
    assert backend.searches == after_first      # no second retrieval in the same session


# ---- the local embedder that lets Graphiti run on one OpenRouter key ----

def test_embed_text_is_deterministic_and_unit_length():
    from graphiti_embedder import EMBEDDING_DIM, embed_text
    a = embed_text("I book 5-star hotels only")
    b = embed_text("I book 5-star hotels only")
    assert a == b and len(a) == EMBEDDING_DIM
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9


def test_embed_text_ranks_related_text_higher():
    from graphiti_embedder import cosine, embed_text
    query = embed_text("hotel star rating preference")
    close = embed_text("I book 5-star hotels only")
    far = embed_text("the dead letter queue rejected a booking message")
    assert cosine(query, close) > cosine(query, far)


def test_embed_text_handles_empty_input():
    from graphiti_embedder import embed_text
    assert set(embed_text("")) == {0.0}


def test_build_embedder_openai_requires_a_key():
    import pytest as _pytest
    from config import Settings
    from graphiti_embedder import build_embedder
    settings = Settings(_env_file=None, GRAPHITI_EMBEDDER="openai", GRAPHITI_EMBEDDER_API_KEY=None,
                        YARVEL_SECRET=None, YARVEL_ORG_ID=None)
    with _pytest.raises(Exception):
        build_embedder(settings)


def test_settings_default_to_local_embedder_and_project_model():
    from config import Settings
    s = Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None)
    assert s.graphiti_embedder == "local"
    assert s.graphiti_llm_model == "anthropic/claude-haiku-4.5"
    assert s.memory_backend == "local"
