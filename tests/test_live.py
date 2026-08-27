"""Tests that need something real running. Each one is gated on its own, so a
missing key skips only what depends on it.

    RUN_LIVE=1 pytest tests/test_live.py            Yarvel + OpenRouter
    RUN_GRAPHITI=1 pytest tests/test_live.py        the graph in the container
"""
import os
from datetime import datetime, timezone

import pytest

live = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1",
                          reason="set RUN_LIVE=1")
needs_yarvel = pytest.mark.skipif(
    not (os.environ.get("YARVEL_SECRET") and os.environ.get("YARVEL_ORG_ID")),
    reason="set YARVEL_SECRET and YARVEL_ORG_ID")
needs_openrouter = pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"),
                                      reason="set OPENROUTER_API_KEY")
needs_graph = pytest.mark.skipif(os.environ.get("RUN_GRAPHITI") != "1",
                                 reason="set RUN_GRAPHITI=1 with FalkorDB reachable")


@live
@needs_yarvel
@pytest.mark.asyncio
async def test_destination_lookup_reaches_yarvel():
    import hotel_tools
    assert isinstance(await hotel_tools.resolve_destination("Jeddah"), list)


@live
@needs_yarvel
@pytest.mark.asyncio
async def test_search_returns_a_session_uuid():
    import hotel_tools
    org = os.environ["YARVEL_ORG_ID"]
    destinations = await hotel_tools.resolve_destination("Jeddah")
    code = next((d.code for d in destinations if d.type == "CITY"), None)
    result = await hotel_tools.start_hotel_search(
        organizationId=org, currency="USD", nationality="SA",
        checkIn="2026-09-01", checkOut="2026-09-05", destinations=[code], adults=2)
    assert result.uuid


@live
@needs_openrouter
@pytest.mark.asyncio
async def test_openrouter_answers():
    from runtime import build_llm
    llm = build_llm()
    try:
        reply = await llm.complete([{"role": "user", "content": "Reply with: ok"}])
        assert reply.content and reply.content.strip()
    finally:
        await llm.aclose()


@needs_graph
@needs_openrouter
@pytest.mark.asyncio
async def test_fact_written_to_the_graph_is_read_back():
    """Writes through Graphiti's extraction pipeline into FalkorDB, then reads it
    back — the path the three production bugs were on. Uses a throwaway user so
    it never touches real memory."""
    from config import Settings
    from memory import GraphitiMemory, GraphitiGraphBackend, graphiti_group_id, user_group

    settings = Settings(_env_file=None, MEMORY_BACKEND="graphiti")
    memory = GraphitiMemory(GraphitiGraphBackend.from_settings(settings))
    username = f"pytest-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"

    await memory.add_user_episode("I book 5-star hotels only.", username,
                                  key="hotel_stars")
    facts = await memory.history("", username=username)
    assert any("5-star" in f.statement for f in facts), "nothing came back from the graph"
    assert all(f.group_id == graphiti_group_id(user_group(username)) for f in facts)
