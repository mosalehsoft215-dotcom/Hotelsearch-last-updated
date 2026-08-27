"""Web console for the TripOn/Rihla agents.

Serves the page and a /chat endpoint that routes to the chosen agent
(hotel search or ops triage), keeping conversation + memory per session. Every
failure is returned as JSON so the page never has to parse an HTML error.

    uvicorn api:app --reload      # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.hotel_search_agent import HotelSearchAgent
from agents.ops_triage_agent import OpsTriageAgent
from config import get_settings
from memory import build_memory
from runtime import AgentContext, build_llm

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


_SESSIONS: dict[str, ChatSession] = {}


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

        llm = build_llm(_settings, model=req.model)
        try:
            result = await _AGENTS[agent_key].run(
                session.ctx, req.message, llm,
                max_iterations=_settings.agent_max_iterations,
                history=session.history or None)
        finally:
            await llm.aclose()

        session.history = result.messages
        return {
            "session_id": session_id,
            "agent": agent_key,
            "output": result.output,
            "verification": {"passed": result.verification.passed,
                             "issues": result.verification.issues},
            "tools_called": [c.name for c in session.ctx.tool_calls],
            "memory": session.ctx.recall_all(),
            "remembered": session.ctx.memory_context,
            "model": getattr(llm, "model", None),
        }
    except Exception as exc:  # never 500 — the page expects JSON
        logger.exception("chat failed")
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


@app.get("/health")
async def health() -> dict[str, Any]:
    """Is the service up, and which memory backend did it start with."""
    return {"ok": True, "memory_backend": _settings.memory_backend,
            "graphiti_embedder": _settings.graphiti_embedder,
            "model": _settings.openrouter_model}


@app.get("/models")
async def models() -> dict[str, Any]:
    """The models the page can switch between. First one is the default."""
    return {"models": [o["model"] for o in _settings.model_options()]}
