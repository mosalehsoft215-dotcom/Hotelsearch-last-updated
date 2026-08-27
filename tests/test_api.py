"""The /chat web endpoint, driven by a scripted LLM and mocked Hasura."""
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
