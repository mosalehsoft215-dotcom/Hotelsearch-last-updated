"""The /chat web endpoint, driven by a scripted LLM and mocked Hasura."""
import json

import pytest
from fastapi.testclient import TestClient

import api
from runtime import LLMResponse, LLMToolCall

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete(self, messages, tools=None):
        return self._responses.pop(0)

    async def aclose(self):
        return None


def _hotel(name, price):
    return {"hotelName": name, "hotelCode": name[:3].upper(), "available": True,
            "price": {"totalPrice": price, "currency": "USD"},
            "location": {"city": "Metroville", "country": "AE"}, "categoryCode": "5"}


@pytest.fixture
def client(monkeypatch, fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("Alpha", 90)]}
    scripted = ScriptedLLM([
        LLMResponse(tool_calls=[LLMToolCall("c1", "search_hotel_availability",
            {"city": "Metroville", "checkIn": "2026-08-15", "checkOut": "2026-08-20", "adults": 2})]),
        LLMResponse(content="Cheapest is Alpha at 90 USD total."),
    ])
    monkeypatch.setattr(api, "build_llm", lambda *_a, **_k: scripted)
    api._SESSIONS.clear()
    return TestClient(api.app)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200 and "TripOn" in r.text and "Ops Triage" in r.text


def test_chat_runs_agent_and_returns_verification(client):
    r = client.post("/chat", json={
        "message": "Find me a hotel in Metroville from Aug 15 to 20 for 2 adults", "org_id": ORG,
    })
    data = r.json()
    assert data["output"] == "Cheapest is Alpha at 90 USD total."
    assert data["verification"]["passed"] is True
    assert "search_hotel_availability" in data["tools_called"]
    assert data["memory"]["hotel_search_session_id"] == "U1"
    assert data["session_id"]


def test_chat_missing_org_id_is_reported(monkeypatch, client):
    monkeypatch.setattr(api._settings, "yarvel_org_id", None, raising=False)
    r = client.post("/chat", json={"message": "hi"})
    assert "org_id" in r.json().get("error", "")


def test_chat_routes_to_triage(monkeypatch, fake_hasura):
    from fastapi.testclient import TestClient
    import api
    from runtime import LLMResponse, LLMToolCall
    from tests.conftest import Seq

    fake_hasura.responses["Core_BookingQueueStatus"] = Seq([
        [{"Status": "failed"}, {"Status": "complete"}],
        [{"Id": 1, "MessageId": "M1", "Status": "failed", "OperationType": "hotel_book",
          "ErrorMessage": "Supplier timeout", "TransactionId": "T1", "CreatedAt": "2026-08-01"}],
    ])
    script = [LLMResponse(tool_calls=[LLMToolCall("c1", "get_queue_summary", {})]),
              LLMResponse(tool_calls=[LLMToolCall("c2", "get_failed_messages", {})]),
              LLMResponse(content="1 failed: supplier_timeout. Recommend escalate to supplier.")]

    class _LLM:
        def __init__(self): self.r = list(script)
        async def complete(self, m, tools=None): return self.r.pop(0)
        async def aclose(self): pass

    monkeypatch.setattr(api, "build_llm", lambda *a, **k: _LLM())
    api._SESSIONS.clear()
    r = TestClient(api.app).post("/chat", json={"message": "triage", "agent": "triage", "org_id": ORG})
    d = r.json()
    assert d["agent"] == "triage"
    assert d["verification"]["passed"], d["verification"]
    assert "get_queue_summary" in d["tools_called"]


def test_memory_endpoint_and_chat_stores_across_sessions(monkeypatch, fake_hasura):
    """The console shares one durable store, so a preference stated in one
    session is injected into the next."""
    from fastapi.testclient import TestClient
    import api
    from runtime import LLMResponse, LLMToolCall

    prompts = []

    class LLM:
        def __init__(self, script): self.script = list(script)
        async def complete(self, messages, tools=None):
            prompts.append(messages[0]["content"])
            return self.script.pop(0)
        async def aclose(self): pass

    # turn 1: the model stores a preference, then answers
    script1 = [LLMResponse(tool_calls=[LLMToolCall("c1", "remember_preference",
                  {"statement": "I book 5-star only.", "key": "hotel_stars"})]),
               LLMResponse(content="Noted.")]
    monkeypatch.setattr(api, "build_llm", lambda *a, **k: LLM(script1))
    api._SESSIONS.clear()
    client = TestClient(api.app)
    first = client.post("/chat", json={"message": "I book 5-star only", "agent": "hotel",
                                      "org_id": ORG, "username": "u1"}).json()
    assert "remember_preference" in first["tools_called"]

    # the read-only view shows the stored fact
    view = client.get("/memory", params={"username": "u1", "org_id": ORG}).json()
    assert any("5-star" in f["statement"] and f["current"] for f in view["facts"])

    # turn 2: a brand-new session gets it injected into the system prompt
    monkeypatch.setattr(api, "build_llm", lambda *a, **k: LLM([LLMResponse(content="ok")]))
    api._SESSIONS.clear()
    second = client.post("/chat", json={"message": "find me a hotel", "agent": "hotel",
                                       "org_id": ORG, "username": "u1"}).json()
    assert "5-star" in prompts[-1]
    assert second["remembered"] and "5-star" in second["remembered"]


# --------------------------------------------------------------------------
# The console must report what a turn did, and be able to show the enrichment
# index and a delegation without either one being taken on trust.
# --------------------------------------------------------------------------

def _two_turn_client(monkeypatch):
    """A session whose first turn fetches from the web and whose second reads
    only the index — the exact pair the RAG demo depends on."""
    turns = [
        [LLMResponse(tool_calls=[LLMToolCall("a", "enrich_destination", {"city": "Jeddah"})]),
         LLMResponse(content="Fetched.")],
        [LLMResponse(tool_calls=[LLMToolCall("b", "search_enrichment",
                                            {"question": "how warm will it be"})]),
         LLMResponse(content="29-34 C.")],
    ]
    seen = {"n": 0}

    def build(*_a, **_k):
        llm = ScriptedLLM(turns[seen["n"]])
        seen["n"] += 1
        return llm

    monkeypatch.setattr(api, "build_llm", build)
    api._SESSIONS.clear()
    return TestClient(api.app)


def test_tools_called_is_this_turn_not_the_session(monkeypatch):
    """ctx.tool_calls accumulates for the life of the session so verify() can still
    see a write tool called on turn 1. The page needs the turn, or the chip row
    claims the second question went to the web when it did not."""
    client = _two_turn_client(monkeypatch)
    first = client.post("/chat", json={"message": "fetch Jeddah weather", "org_id": ORG}).json()
    second = client.post("/chat", json={"message": "how warm will it be", "org_id": ORG,
                                        "session_id": first["session_id"]}).json()

    assert first["tools_called"] == ["enrich_destination"]
    assert second["tools_called"] == ["search_enrichment"]
    assert second["tools_called_session"] == ["enrich_destination", "search_enrichment"]


def test_verify_still_sees_the_whole_session(monkeypatch, fake_hasura):
    """The narrowing above is display only. A banned tool called on an early turn
    must still fail verification on a later one."""
    from tests.conftest import Seq
    fake_hasura.responses["Core_BookingQueueStatus"] = Seq([[{"Status": "failed"}], []])
    turns = [
        [LLMResponse(tool_calls=[LLMToolCall("a", "get_queue_summary", {})]),
         LLMResponse(content="queue read")],
        [LLMResponse(content="nothing this turn")],
    ]
    seen = {"n": 0}

    def build(*_a, **_k):
        llm = ScriptedLLM(turns[seen["n"]]); seen["n"] += 1; return llm

    monkeypatch.setattr(api, "build_llm", build)
    api._SESSIONS.clear()
    client = TestClient(api.app)
    first = client.post("/chat", json={"message": "triage", "agent": "triage", "org_id": ORG}).json()
    second = client.post("/chat", json={"message": "and now?", "agent": "triage", "org_id": ORG,
                                        "session_id": first["session_id"]}).json()
    assert second["tools_called"] == []
    assert second["tools_called_session"] == ["get_queue_summary"]


def test_enrichment_endpoint_reports_size_and_never_fetches(monkeypatch):
    client = TestClient(api.app)
    empty = client.get("/enrichment").json()
    assert "indexed_claims" in empty and empty["matches"] == []

    called = {"n": 0}

    def fake_search(question, limit=5, subject=None, domain=None, entityType=None,
                    entityRef=None, minScore=None):
        called["n"] += 1

        async def _run():
            return {"question": question, "indexed_claims": 5, "note": None,
                    "matches": [{"domain": "weather", "field": "forecast_2026-09-01",
                                 "value": "29.1-34.6 C", "status": "single_source",
                                 "entity_type": "city", "entity_ref": "Jeddah", "match": 0.73,
                                 "sources": [{"url": "https://open-meteo.com/x", "title": "t",
                                              "tier": "official"}]}]}
        return _run()

    monkeypatch.setattr(api, "search_enrichment", fake_search)
    hit = client.get("/enrichment", params={"q": "how warm will it be"}).json()
    assert called["n"] == 1
    assert hit["matches"][0]["domain"] == "weather"
    assert hit["matches"][0]["match"] == 0.73
    assert hit["matches"][0]["sources"][0]["url"].startswith("https://")


def test_enrichment_endpoint_returns_error_not_html_for_a_bad_domain():
    client = TestClient(api.app)
    d = client.get("/enrichment", params={"q": "x", "domain": "not_a_domain"}).json()
    assert "error" in d and "not_a_domain" in d["error"]


def test_delegate_runs_the_child_and_leaves_the_parent_alone(monkeypatch, fake_hasura):
    """The isolation claim, through the endpoint the console calls."""
    from tests.conftest import Seq
    fake_hasura.responses["Core_BookingQueueStatus"] = Seq([
        [{"Status": "failed"}, {"Status": "complete"}],
        [{"Id": 1, "MessageId": "M1", "Status": "failed", "OperationType": "hotel_book",
          "ErrorMessage": "Supplier timeout", "TransactionId": "T1", "CreatedAt": "2026-08-01"}],
    ])
    parent_script = [LLMResponse(content="hello")]
    child_script = [LLMResponse(tool_calls=[LLMToolCall("c1", "get_queue_summary", {})]),
                    LLMResponse(tool_calls=[LLMToolCall("c2", "get_failed_messages", {})]),
                    LLMResponse(content="1 failed: supplier_timeout.")]
    scripts = [parent_script, child_script]
    seen = {"n": 0}

    def build(*_a, **_k):
        llm = ScriptedLLM(scripts[seen["n"]]); seen["n"] += 1; return llm

    monkeypatch.setattr(api, "build_llm", build)
    api._SESSIONS.clear()
    client = TestClient(api.app)
    first = client.post("/chat", json={"message": "hi", "org_id": ORG}).json()

    parent_ctx = api._SESSIONS[f"{first['session_id']}:hotel"].ctx
    parent_ctx.remember("private_note", "do not leak this")

    d = client.post("/delegate", json={"session_id": first["session_id"],
                                       "brief": "summarise the failed booking queue"}).json()

    assert d["handover"]["agent"] == "ops_triage_agent"
    assert "get_failed_messages" in d["handover"]["tools_used"]
    # names travel back, payloads do not
    assert all(isinstance(t, str) for t in d["handover"]["tools_used"])
    assert d["parent_unchanged"] is True
    assert d["parent_before"] == d["parent_after"]
    assert parent_ctx.recall("private_note") == "do not leak this"
    assert [c.name for c in parent_ctx.tool_calls] == []


def test_delegate_without_a_session_says_so():
    api._SESSIONS.clear()
    d = TestClient(api.app).post("/delegate", json={"session_id": "nope", "brief": "x"}).json()
    assert "No such session" in d["error"]


def test_delegate_finds_the_session_whatever_agent_owns_it(monkeypatch, fake_hasura):
    """The session id is keyed per agent. Delegating from the ops tab must not
    fail just because no hotel session exists under that id."""
    from tests.conftest import Seq
    fake_hasura.responses["Core_BookingQueueStatus"] = Seq([[{"Status": "failed"}], [], [], []])
    scripts = [[LLMResponse(content="triaged")],
               [LLMResponse(content="handed back")]]
    seen = {"n": 0}

    def build(*_a, **_k):
        llm = ScriptedLLM(scripts[seen["n"]]); seen["n"] += 1; return llm

    monkeypatch.setattr(api, "build_llm", build)
    api._SESSIONS.clear()
    client = TestClient(api.app)
    first = client.post("/chat", json={"message": "triage", "agent": "triage",
                                       "org_id": ORG}).json()
    assert f"{first['session_id']}:hotel" not in api._SESSIONS

    d = client.post("/delegate", json={"session_id": first["session_id"],
                                       "brief": "look again"}).json()
    assert "error" not in d, d
    assert d["parent_agent"] == "triage"
    assert d["parent_unchanged"] is True


def test_the_agent_s_own_answers_survive_into_the_next_turn(monkeypatch, fake_hasura):
    """The loop used to break out on the final response without appending it, so
    session.history held [system, user, user] and the agent had no record of what
    it had just said. "Which one was cheapest?" had nothing to refer back to."""
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("Alpha", 90)]}
    seen_second_turn = {}
    turns = [[LLMResponse(content="Cheapest is Alpha at 90 USD.")],
             [LLMResponse(content="Alpha, as I said.")]]
    n = {"i": 0}

    class LLM:
        def __init__(self, script): self.script = list(script)
        async def complete(self, messages, tools=None):
            # snapshot: `messages` is mutated after this returns
            seen_second_turn["messages"] = [dict(m) for m in messages]
            return self.script.pop(0)
        async def aclose(self): pass

    def build(*_a, **_k):
        llm = LLM(turns[n["i"]]); n["i"] += 1; return llm

    monkeypatch.setattr(api, "build_llm", build)
    api._SESSIONS.clear()
    client = TestClient(api.app)
    first = client.post("/chat", json={"message": "find a hotel", "org_id": ORG}).json()
    client.post("/chat", json={"message": "which one was cheapest?", "org_id": ORG,
                               "session_id": first["session_id"]})

    roles = [m["role"] for m in seen_second_turn["messages"]]
    assert roles == ["system", "user", "assistant", "user"], roles
    assert "Cheapest is Alpha at 90 USD." in seen_second_turn["messages"][2]["content"]


def test_chat_returns_json_even_when_session_memory_holds_something_odd(monkeypatch, client):
    """FastAPI encodes after the handler returns, so an unencodable value used to
    escape the except and reach the page as a plain-text 500."""
    r = client.post("/chat", json={
        "message": "Find me a hotel in Metroville from Aug 15 to 20 for 2 adults",
        "org_id": ORG})
    sid = r.json()["session_id"]
    api._SESSIONS[f"{sid}:hotel"].ctx.remember("odd", object())
    scripted = ScriptedLLM([LLMResponse(content="fine")])
    monkeypatch.setattr(api, "build_llm", lambda *_a, **_k: scripted)
    second = client.post("/chat", json={"message": "again", "org_id": ORG, "session_id": sid})
    assert second.status_code == 200
    assert second.headers["content-type"].startswith("application/json")


@pytest.mark.asyncio
async def test_one_turn_at_a_time_per_session(monkeypatch, fake_hasura):
    """Two requests for the same session must not run against the same history.

    Without the per-session lock both turns read the transcript as it was before
    either started, and whichever finished last overwrote the other's — one turn
    silently lost. The scripted model here parks inside the first turn until the
    second request has been issued, which is the interleaving that used to lose it.
    """
    import asyncio

    import httpx

    started = asyncio.Event()
    release = asyncio.Event()
    turns = [[LLMResponse(content="first")], [LLMResponse(content="second")]]
    built = {"n": 0}

    class SlowLLM:
        def __init__(self, script, park):
            self.script = list(script)
            self.park = park

        async def complete(self, messages, tools=None):
            if self.park:
                started.set()
                await release.wait()
            return self.script.pop(0)

        async def aclose(self):
            return None

    def build(*_a, **_k):
        llm = SlowLLM(turns[built["n"]], park=built["n"] == 0)
        built["n"] += 1
        return llm

    monkeypatch.setattr(api, "build_llm", build)
    api._SESSIONS.clear()

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        sid = "concurrent"
        one = asyncio.create_task(http.post("/chat", json={
            "message": "first question", "org_id": ORG, "session_id": sid}))
        await started.wait()
        two = asyncio.create_task(http.post("/chat", json={
            "message": "second question", "org_id": ORG, "session_id": sid}))
        await asyncio.sleep(0)          # let the second request reach the lock
        release.set()
        first, second = await asyncio.gather(one, two)

    assert first.json()["output"] == "first"
    assert second.json()["output"] == "second"
    history = api._SESSIONS[f"{sid}:hotel"].history
    contents = [m.get("content") for m in history]
    assert "first question" in contents and "second question" in contents
    assert "first" in contents and "second" in contents


def test_models_reports_the_host_for_each_model():
    """The page keeps the conversation across models on one host and starts a
    fresh one across hosts, because a transcript carries provider-specific
    fields (Gemini's thought_signature) the next provider rejects."""
    client = TestClient(api.app)
    body = client.get("/models").json()
    assert body["models"], "at least the default is configured"
    assert set(body["hosts"]) == set(body["models"])
    assert all(h and "/" not in h for h in body["hosts"].values()), body["hosts"]


def test_a_turn_is_recorded_without_the_message_text(monkeypatch, client, tmp_path):
    """The log has to be keepable: names, counts and outcomes, no customer text
    and no tool payloads."""
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(api._settings, "run_log_path", str(log), raising=False)
    client.post("/chat", json={"message": "Find me a hotel in Metroville", "org_id": ORG})
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["agent"] == "hotel" and row["verified"] is True
    assert "search_hotel_availability" in row["tools"]
    assert row["message_chars"] == len("Find me a hotel in Metroville")
    assert isinstance(row["ms"], int) and row["ts"]
    assert "Metroville" not in lines[0], "no message text in the log"


def test_no_log_file_is_written_when_the_path_is_unset(monkeypatch, client, tmp_path):
    monkeypatch.setattr(api._settings, "run_log_path", None, raising=False)
    client.post("/chat", json={"message": "hi", "org_id": ORG})
    assert list(tmp_path.iterdir()) == []
