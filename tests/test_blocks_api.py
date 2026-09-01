"""The wire shape: blocks travel to the client, and answer-only clients cannot tell.

Uses the same scripted-LLM + mocked-Hasura harness as tests/test_api.py, so this
is the real /chat path rather than a hand-built response.
"""
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
            "location": {"city": "Metroville", "country": "AE"}, "categoryCode": "5",
            "board": "Room Only", "pricePerNight": price / 5,
            "optionRefId": "33!~|a0!~|SECRET",
            "cancelPolicy": {"refundable": True, "description": "Free until 10 Aug"}}


def _searching_client(monkeypatch, fake_hasura, answer):
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "hotels": [_hotel("Alpha", 90), _hotel("Beta", 120)]}
    scripted = ScriptedLLM([
        LLMResponse(tool_calls=[LLMToolCall("c1", "search_hotel_availability",
            {"city": "Metroville", "checkIn": "2026-08-15", "checkOut": "2026-08-20",
             "adults": 2})]),
        LLMResponse(content=answer),
    ])
    monkeypatch.setattr(api, "build_llm", lambda *_a, **_k: scripted)
    return TestClient(api.app)


def _plain_client(monkeypatch, fake_hasura, answer):
    monkeypatch.setattr(api, "build_llm",
                        lambda *_a, **_k: ScriptedLLM([LLMResponse(content=answer)]))
    return TestClient(api.app)


# ---- a hotel search puts cards on the wire ----

def test_a_hotel_search_returns_blocks_beside_the_answer(monkeypatch, fake_hasura):
    client = _searching_client(monkeypatch, fake_hasura,
                              "**Two options.** Alpha is the cheaper of the two.")
    body = client.post("/chat", json={"message": "find a hotel in Metroville",
                                      "org_id": ORG}).json()

    assert body["output"].startswith("**Two options.**")
    assert "blocks" in body
    kinds = [b["type"] for b in body["blocks"]]
    assert kinds == ["hotel_option", "hotel_option"]
    assert [b["hotel_name"] for b in body["blocks"]] == ["Alpha", "Beta"]
    first = body["blocks"][0]
    assert (first["total_price"], first["currency"]) == (90.0, "USD")
    assert first["stars"] == 5.0 and first["refundable"] is True


def test_the_answer_carries_no_json_and_the_blocks_carry_no_internals(monkeypatch,
                                                                     fake_hasura):
    client = _searching_client(monkeypatch, fake_hasura, "Alpha is cheapest at 90 USD.")
    body = client.post("/chat", json={"message": "find a hotel", "org_id": ORG}).json()

    assert "{" not in body["output"] and "hotel_name" not in body["output"]
    wire = str(body["blocks"])
    for internal in ("optionRefId", "33!~|", "SECRET", "hotelCode", "ALP", "U1"):
        assert internal not in wire, internal


# ---- an answer-only client cannot tell this exists ----

def test_a_plain_conversational_turn_omits_the_field_entirely(monkeypatch, fake_hasura):
    client = _plain_client(monkeypatch, fake_hasura, "Riyadh is hot in September.")
    body = client.post("/chat", json={"message": "hello", "org_id": ORG}).json()

    assert body["output"] == "Riyadh is hot in September."
    assert "blocks" not in body, "absent, not null, when there is nothing structured"
    assert "sources" not in body


def test_an_existing_client_reading_only_output_is_unaffected(monkeypatch, fake_hasura):
    """Requirement 7. The fields that were there before are all still there, with
    the same names, whether or not blocks came along."""
    was = {"session_id", "agent", "output", "verification", "tools_called",
           "tools_called_session", "memory", "remembered", "model"}

    plain = _plain_client(monkeypatch, fake_hasura, "Just prose.").post(
        "/chat", json={"message": "hi", "org_id": ORG}).json()
    assert was <= set(plain), was - set(plain)
    assert set(plain) == was, "a prose turn's shape is byte-for-byte what it was"

    withcards = _searching_client(monkeypatch, fake_hasura, "Alpha is cheapest.").post(
        "/chat", json={"message": "find a hotel", "org_id": ORG}).json()
    assert was <= set(withcards), was - set(withcards)
    assert set(withcards) - was == {"blocks"}
    # The one thing an old client reads still reads the same way.
    assert isinstance(withcards["output"], str) and withcards["output"]


def test_the_verification_and_tool_chips_are_unchanged(monkeypatch, fake_hasura):
    body = _searching_client(monkeypatch, fake_hasura, "Alpha is cheapest at 90 USD.").post(
        "/chat", json={"message": "find a hotel", "org_id": ORG}).json()
    assert body["tools_called"] == ["search_hotel_availability"]
    assert body["verification"]["passed"] is True, body["verification"]["issues"]


# ---- the renderer is served next to the page ----

def test_the_render_module_is_served(monkeypatch, fake_hasura):
    client = _plain_client(monkeypatch, fake_hasura, "x")
    res = client.get("/chat_render.js")
    assert res.status_code == 200
    assert "javascript" in res.headers["content-type"]
    assert "renderAssistantMessage" in res.text
    assert "noopener noreferrer" in res.text


def test_the_page_loads_the_render_module(monkeypatch, fake_hasura):
    client = _plain_client(monkeypatch, fake_hasura, "x")
    page = client.get("/").text
    assert '<script src="/chat_render.js"></script>' in page
    assert "renderAssistantMessage" in page, "the page actually calls the renderer"
