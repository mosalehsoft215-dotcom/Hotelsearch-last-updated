"""Demo: what the durable memory layer buys us.

Runs on the local backend, so it needs no database and no API keys:

    python demo_memory.py

Set MEMORY_BACKEND=graphiti in .env to run the same four scenes against
FalkorDB through graphiti_core.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from config import get_settings
from memory import build_memory

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"
USER = "m.saleh"
OPS_AGENT = "ops_triage_agent"


def scene(number: int, title: str) -> None:
    print(f"\n{'=' * 68}\n{number}. {title}\n{'=' * 68}")


def show(label: str, value) -> None:
    print(f"  {label}: {value}")


async def main() -> None:
    settings = get_settings()
    memory = build_memory(settings)
    print(f"backend: {settings.memory_backend}   top_k: {memory.top_k}")

    march = datetime(2026, 3, 2, tzinfo=timezone.utc)
    july = datetime(2026, 7, 20, tzinfo=timezone.utc)

    scene(1, "A preference survives the session")
    await memory.add_user_episode(
        "I book 4-star hotels or better, never below that.", USER,
        key="hotel_stars", reference_time=march)
    print("  session 1 — user said it, agent stored it")
    context = await memory.get_context("find a hotel for a client", username=USER, org_id=ORG)
    print("  session 2 — brand new session, nothing in SessionMemory:")
    for line in (context or "").splitlines():
        print(f"    {line}")

    scene(2, "The preference changes — the old one is kept, not deleted")
    await memory.add_user_episode(
        "Since the promotion I book 5-star only.", USER,
        key="hotel_stars", reference_time=july)
    current = await memory.get_context("hotel standard", username=USER)
    print("  what the agent is told now:")
    for line in (current or "").splitlines():
        print(f"    {line}")
    print("  full record, superseded facts included:")
    for fact in await memory.history("hotel", username=USER):
        state = "current" if fact.current else f"was true until {fact.invalid_at:%Y-%m-%d}"
        print(f"    - {fact.statement}  [{state}]")

    scene(3, "Org default applies, the user's own choice wins")
    await memory.add_org_episode(
        "Agency default board is room only.", ORG, key="board", reference_time=march)
    await memory.add_org_episode(
        "Agency currency is USD.", ORG, key="currency", reference_time=march)
    await memory.add_user_episode(
        "I want breakfast included on every booking.", USER,
        key="board", reference_time=july)
    merged = await memory.get_context("board and currency", username=USER, org_id=ORG)
    print("  merged view given to the agent (user overrides org on board):")
    for line in (merged or "").splitlines():
        print(f"    {line}")

    scene(4, "An ops failure signature is recognised next session")
    signature = "QuotationToBookingV2:validation_error"
    first = await memory.is_known_ops_pattern(signature, OPS_AGENT)
    show("session 1 — known already?", first)
    await memory.add_ops_episode(signature, OPS_AGENT, key=signature)
    print("  recorded")
    second = await memory.is_known_ops_pattern(signature, OPS_AGENT)
    show("session 2 — known already?", second)
    print("  so the report says recurring instead of flagging it as new")

    scene(5, "What is actually stored")
    backend = memory.backend
    episodes = getattr(backend, "episodes", None)
    facts = getattr(backend, "facts", None)
    if episodes is None:
        print("  (graphiti backend — inspect the graph in FalkorDB)")
        return
    show("episodes kept verbatim", len(episodes))
    show("facts", f"{sum(1 for f in facts if f.current)} current, "
                  f"{sum(1 for f in facts if not f.current)} superseded")
    print("  groups:", sorted({e.group_id for e in episodes}))


if __name__ == "__main__":
    asyncio.run(main())
