"""Web console for the TripOn/Rihla agents.

Serves the page and a /chat endpoint that routes to the chosen agent
(hotel search or ops triage), keeping conversation + memory per session. Every
failure is returned as JSON so the page never has to parse an HTML error.

    uvicorn api:app --reload      # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.hotel_search_agent import HotelSearchAgent
from agents.ops_triage_agent import OpsTriageAgent
from config import get_settings
from memory import build_memory
from runtime import AgentContext, build_llm, delegate
from web_tools import index_stats, search_enrichment

logger = logging.getLogger("tripon.agents.api")

app = FastAPI(title="TripOn Agents")
_settings = get_settings()
_AGENTS = {"hotel": HotelSearchAgent(), "triage": OpsTriageAgent()}
_MEMORY = build_memory(_settings)   # durable across sessions and page reloads
_CHAT_HTML = (Path(__file__).parent / "chat_ui.html").read_text(encoding="utf-8")


@dataclass
class ChatSession:
    ctx: AgentContext
    history: list[dict[str, Any]] = field(default_factory=list)
    # One turn at a time per session. Two concurrent /chat calls would both read
    # `history` and the later writer would drop the earlier turn; a /delegate
    # snapshotting `ctx` around a chat that is still running would straddle that
    # chat's own tool calls and report the parent as changed with no isolation
    # failure behind it. Send and Delegate take this same lock.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_SESSIONS: dict[str, ChatSession] = {}


def record_turn(**fields: Any) -> None:
    """One line per turn: which model on which host, what it called, whether
    verify passed, how long it took.

    No message text and no tool payloads — lengths and names only, so the log
    can be kept and read without carrying customer data. Always logged; also
    appended to HOTELS_RUN_LOG when that is set.
    """
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    logger.info("agent_turn", extra={"turn": record})
    path = _settings.run_log_path
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        # A log that cannot be written must not take the turn with it.
        logger.warning("run log unwritable: %s", exc)


class ChatRequest(BaseModel):
    message: str
    agent: str = "hotel"
    session_id: str | None = None
    org_id: str | None = None
    username: str | None = None
    model: str | None = None
    currency: str | None = None
    nationality: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _CHAT_HTML


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    session_id = req.session_id or uuid.uuid4().hex
    agent_key = req.agent if req.agent in _AGENTS else "hotel"
    key = f"{session_id}:{agent_key}"
    try:
        org_id = req.org_id or _settings.yarvel_org_id
        if not org_id:
            return {"error": "No org_id. Pass one, or set YARVEL_ORG_ID in .env.",
                    "session_id": session_id}
        session = _SESSIONS.get(key)
        if session is None:
            session = ChatSession(ctx=AgentContext(
                org_id=org_id,
                username=req.username or _settings.yarvel_username or "demo_user",
                memory=_MEMORY,
                currency=req.currency or _settings.default_currency,
                nationality=req.nationality or _settings.default_nationality,
            ))
            _SESSIONS[key] = session

        started = time.perf_counter()
        async with session.lock:
            llm = build_llm(_settings, model=req.model)
            # ctx.tool_calls accumulates for the life of the session, because
            # verify() must still see a write tool called on turn 1 when it runs
            # on turn 5. The page wants the opposite: what this turn did. Mark
            # the boundary and slice.
            before = len(session.ctx.tool_calls)
            try:
                result = await _AGENTS[agent_key].run(
                    session.ctx, req.message, llm,
                    max_iterations=_settings.agent_max_iterations,
                    history=session.history or None)
            finally:
                await llm.aclose()

            session.history = result.messages

        record_turn(session_id=session_id, agent=agent_key,
                    model=getattr(llm, "model", None), host=getattr(llm, "host", None),
                    tools=[c.name for c in session.ctx.tool_calls[before:]],
                    verified=result.verification.passed,
                    issues=result.verification.issues,
                    message_chars=len(req.message), output_chars=len(result.output),
                    ms=round((time.perf_counter() - started) * 1000))
        # Serialise inside the try. FastAPI encodes the return value after the
        # handler exits, so an unencodable value in ctx.recall_all() would escape
        # this except and reach the page as a plain-text 500.
        return jsonable_encoder({
            "session_id": session_id,
            "agent": agent_key,
            "output": result.output,
            "verification": {"passed": result.verification.passed,
                             "issues": result.verification.issues},
            "tools_called": [c.name for c in session.ctx.tool_calls[before:]],
            "tools_called_session": [c.name for c in session.ctx.tool_calls],
            "memory": session.ctx.recall_all(),
            "remembered": session.ctx.memory_context,
            "model": getattr(llm, "model", None),
        })
    except Exception as exc:  # never 500 — the page expects JSON
        logger.exception("chat failed")
        record_turn(session_id=session_id, agent=agent_key, model=req.model,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    message_chars=len(req.message))
        return {"error": f"{type(exc).__name__}: {exc}", "session_id": session_id}


@app.get("/memory")
async def memory_view(query: str = "", username: str | None = None,
                      org_id: str | None = None) -> dict[str, Any]:
    """What the durable layer holds — current facts, superseded ones, and the
    context the agent would be given. Read-only."""
    user = username or _settings.yarvel_username or "demo_user"
    org = org_id or _settings.yarvel_org_id
    try:
        # Pass the query through as given. Substituting a word like "preferences"
        # for the empty case looks harmless but is a lexical probe: against the
        # local embedder it matches nothing, so the panel showed no context while
        # the graph held facts. Empty means "list what is there".
        context = await _MEMORY.get_context(query, username=user, org_id=org)
        facts = await _MEMORY.history(query, username=user, org_id=org)
        return {
            "backend": _settings.memory_backend,
            "username": user,
            "context": context,
            "facts": [{"statement": f.statement, "group": f.group_id, "key": f.key,
                       "current": f.current,
                       "valid_at": f.valid_at.isoformat() if f.valid_at else None,
                       "invalid_at": f.invalid_at.isoformat() if f.invalid_at else None}
                      for f in facts],
        }
    except Exception as exc:
        logger.exception("memory view failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/enrichment")
async def enrichment_view(q: str = "", limit: int = 5, domain: str | None = None,
                          entityType: str | None = None, entityRef: str | None = None,
                          minScore: float | None = None) -> dict[str, Any]:
    """What the enrichment index holds, and what a plain question retrieves from it.

    Read-only, and it never fetches: this is the index being queried, not the web.
    An empty q reports the size so the panel can show a count before anyone searches.
    """
    try:
        if not q.strip():
            return {"question": "", "indexed_claims": index_stats(), "matches": [],
                    "note": None}
        return await search_enrichment(q, limit=limit, domain=domain,
                                       entityType=entityType, entityRef=entityRef,
                                       minScore=minScore)
    except ValueError as exc:            # unknown domain / entityType
        return {"error": str(exc), "indexed_claims": index_stats(), "matches": []}
    except Exception as exc:
        logger.exception("enrichment view failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


class DelegateRequest(BaseModel):
    session_id: str
    brief: str
    agent: str = "triage"
    model: str | None = None


def _snapshot(ctx: AgentContext) -> dict[str, Any]:
    """The two things a caller could lose to a helper: its tool results and its
    scratchpad. Taken before and after so the response carries its own evidence."""
    return {"tool_calls": [c.name for c in ctx.tool_calls],
            "session_keys": sorted(ctx.recall_all())}


@app.post("/delegate")
async def delegate_view(req: DelegateRequest) -> dict[str, Any]:
    """Hand a brief to a second agent on behalf of the current session.

    Triggered by a person, not by the model — neither agent has a delegation tool,
    and that stays true. What this exposes is the handover itself: the child runs on
    a context built by `for_child`, and the parent is snapshotted either side of the
    call so the isolation is visible in the response rather than asserted in prose.
    """
    agent_key = req.agent if req.agent in _AGENTS else "triage"
    # The caller is whichever agent this session id belongs to. A session id is
    # keyed per agent, so look for any of them rather than assuming "hotel".
    parent_key = next((k for k in _SESSIONS if k.startswith(f"{req.session_id}:")), None)
    session = _SESSIONS.get(parent_key) if parent_key else None
    if session is None:
        return {"error": "No such session. Send a chat message first, then delegate."}
    try:
        # The same lock /chat takes, so the snapshot cannot straddle a turn that
        # is still running and report the parent as changed by its own work.
        async with session.lock:
            parent_before = _snapshot(session.ctx)
            llm = build_llm(_settings, model=req.model)
            try:
                handover = await delegate(_AGENTS[agent_key], req.brief, llm, session.ctx,
                                          max_iterations=_settings.agent_max_iterations)
            finally:
                await llm.aclose()
            parent_after = _snapshot(session.ctx)
        return {
            "handover": handover.to_model(),
            "parent_agent": (parent_key or ":").split(":", 1)[1],
            "parent_before": parent_before,
            "parent_after": parent_after,
            "parent_unchanged": parent_before == parent_after,
        }
    except Exception as exc:
        logger.exception("delegate failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Is the service up, and which memory backend did it start with."""
    return {"ok": True, "memory_backend": _settings.memory_backend,
            "graphiti_embedder": _settings.graphiti_embedder,
            "model": _settings.openrouter_model}


@app.get("/models")
async def models() -> dict[str, Any]:
    """The models the page can switch between. First one is the default.

    `hosts` maps each model to the API host that serves it. The page keeps the
    conversation when switching between models on one host and starts a fresh
    one when the host changes — a transcript carries provider-specific fields
    (Gemini puts a thought_signature on every tool call it makes) that the next
    provider will not accept.
    """
    options = _settings.model_options()
    return {"models": [o["model"] for o in options],
            "hosts": {o["model"]: urlparse(o["base_url"]).netloc or o["base_url"]
                      for o in options}}
