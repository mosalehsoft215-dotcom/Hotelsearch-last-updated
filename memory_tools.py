"""The three memory tools an agent may call.

Each one is an explicit capture or lookup. There is no tool that writes whatever
came back from a supplier — that is the point of the design.
"""
from __future__ import annotations

from typing import Any


async def remember_preference(statement: str, key: str | None = None, *,
                              ctx: Any) -> dict[str, Any]:
    """Store a preference the user stated. A new value for the same key
    supersedes the old one; the old one stays on record."""
    if ctx.memory is None or not ctx.username:
        return {"stored": False, "reason": "no memory or no user in this session"}
    await ctx.memory.add_user_episode(statement, ctx.username, key=key)
    return {"stored": True, "statement": statement, "key": key}


async def recall_preferences(query: str, *, ctx: Any) -> dict[str, Any]:
    """What is already known about this user and organization for a topic."""
    if ctx.memory is None:
        return {"context": None}
    context = await ctx.memory.get_context(query, username=ctx.username, org_id=ctx.org_id)
    return {"context": context}


async def record_ops_pattern(signature: str, *, ctx: Any) -> dict[str, Any]:
    """Record a dead-letter signature and say whether it was already known from
    an earlier session."""
    if ctx.memory is None:
        return {"recorded": False, "known": False, "reason": "no memory in this session"}
    agent = ctx.recall("agent_name") or "ops_triage_agent"
    known = await ctx.memory.is_known_ops_pattern(signature, agent)
    if not known:
        await ctx.memory.add_ops_episode(signature, agent, key=signature)
    return {"recorded": not known, "known": known, "signature": signature}
