import json
import httpx
import pytest
from runtime import OpenRouterLLM, LLMError, build_llm
from config import Settings


def _transport(captured, message, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(status, json={"choices": [{"message": message}]})
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_sends_model_tools_and_auth():
    cap = {}
    llm = OpenRouterLLM(api_key="sk-test", model="anthropic/claude-3-haiku",
                        transport=_transport(cap, {"content": "hi"}))
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    out = await llm.complete([{"role": "user", "content": "hello"}], tools=tools)
    assert cap["headers"]["authorization"] == "Bearer sk-test"
    assert cap["body"]["model"] == "anthropic/claude-3-haiku"
    assert cap["body"]["tools"] == tools and cap["body"]["tool_choice"] == "auto"
    assert out.content == "hi" and out.tool_calls == []
    await llm.aclose()


@pytest.mark.asyncio
async def test_parses_tool_calls():
    msg = {"content": None, "tool_calls": [
        {"id": "c1", "function": {"name": "search_hotel_availability",
                                  "arguments": json.dumps({"city": "Metroville"})}}]}
    llm = OpenRouterLLM(api_key="k", model="m", transport=_transport({}, msg))
    out = await llm.complete([{"role": "user", "content": "find dubai"}])
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "search_hotel_availability" and tc.arguments == {"city": "Metroville"} and tc.id == "c1"
    await llm.aclose()


@pytest.mark.asyncio
async def test_http_error_raises():
    def handler(_):
        return httpx.Response(429, json={"error": "rate limited"})
    llm = OpenRouterLLM(api_key="k", model="m", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await llm.complete([{"role": "user", "content": "x"}])
    await llm.aclose()


def test_missing_key_raises():
    with pytest.raises(LLMError):
        OpenRouterLLM(api_key=None, model="m")


def test_build_llm_from_settings():
    s = Settings(_env_file=None, OPENROUTER_API_KEY="sk-x", OPENROUTER_MODEL="anthropic/claude-3-haiku",
                 YARVEL_SECRET=None, YARVEL_ORG_ID=None)
    llm = build_llm(s)
    assert llm.model == "anthropic/claude-3-haiku"


import pytest
import hotel_tools as srv
import runtime as tool_registry
from runtime import AgentContext

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def test_write_tools_are_not_available():
    for forbidden in ("book_hotel", "cancel_hotel", "send_hotel_cancel_request"):
        assert forbidden not in tool_registry.AVAILABLE_TOOL_NAMES
        assert all(s["function"]["name"] != forbidden for s in tool_registry.TOOL_SPECS)


def test_specs_for_filters_by_name():
    specs = tool_registry.specs_for({"refresh_hotel_price"})
    assert [s["function"]["name"] for s in specs] == ["refresh_hotel_price"]


@pytest.mark.asyncio
async def test_dispatch_injects_context(fake_hasura):
    fake_hasura.responses["Core_HotelBookings"] = []
    ctx = AgentContext(org_id=ORG, currency="GBP", nationality="SA")
    await tool_registry.dispatch("list_hotel_bookings", {"limit": 5}, ctx)
    variables = fake_hasura.calls[0]["variables"]
    assert variables["where"]["OrganizationId"] == {"_eq": ORG}


@pytest.mark.asyncio
async def test_dispatch_returns_jsonable(fake_hasura):
    fake_hasura.responses["Core_HotelBookings_by_pk"] = {"Id": 3, "BookingStatus": "CONFIRMED"}
    ctx = AgentContext(org_id=ORG)
    out = await tool_registry.dispatch("get_hotel_booking", {"Id": 3}, ctx)
    assert isinstance(out, dict) and out["BookingStatus"] == "CONFIRMED"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises():
    with pytest.raises(KeyError):
        await tool_registry.dispatch("book_hotel", {}, AgentContext(org_id=ORG))


@pytest.mark.asyncio
async def test_get_hotel_search_results_available(fake_hasura):
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": []}
    ctx = AgentContext(org_id=ORG)
    out = await tool_registry.dispatch("get_hotel_search_results", {"uuid": "U1"}, ctx)
    assert out["isComplete"] is True
    assert fake_hasura.calls[0]["variables"]["uuid"] == "U1"
    assert fake_hasura.calls[0]["variables"]["organizationId"] == ORG


"""The full loop with a scripted fake LLM (no network) and mocked Hasura."""
import pytest
from runtime import AgentContext
from runtime import LLMResponse, LLMToolCall
from agents.hotel_search_agent import HotelSearchAgent, MEM_SESSION_ID

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_tools = []

    async def complete(self, messages, tools=None):
        self.seen_tools = [t["function"]["name"] for t in (tools or [])]
        return self._responses.pop(0)


def _hotel(name, price):
    return {"hotelName": name, "hotelCode": name[:3].upper(), "available": True,
            "price": {"totalPrice": price, "currency": "USD"},
            "location": {"city": "Metroville", "country": "AE"}, "categoryCode": "5"}


@pytest.mark.asyncio
async def test_loop_calls_tool_then_answers(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("Alpha", 90)]}

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[LLMToolCall("c1", "search_hotel_availability",
            {"city": "Metroville", "checkIn": "2026-08-15", "checkOut": "2026-08-20", "adults": 2})]),
        LLMResponse(content="Cheapest is Alpha at 90 USD total."),
    ])
    ctx = AgentContext(org_id=ORG, currency="USD", nationality="AE")
    result = await HotelSearchAgent().run(ctx, "Find me a hotel in Metroville Aug 15-20 for 2 adults", llm)

    assert result.output == "Cheapest is Alpha at 90 USD total."
    assert [c.name for c in ctx.tool_calls] == ["search_hotel_availability"]
    assert ctx.recall(MEM_SESSION_ID) == "U1"          # memory populated by the loop
    assert result.verification.passed
    # the model was only ever offered read/draft tools
    assert "book_hotel" not in llm.seen_tools and "search_hotel_availability" in llm.seen_tools


@pytest.mark.asyncio
async def test_loop_forces_final_answer_at_cap(fake_hasura):
    fake_hasura.responses["Core_HotelBookings"] = []
    # keeps calling a tool every turn, then a forced no-tools call produces text
    script = [LLMResponse(tool_calls=[LLMToolCall(f"c{i}", "list_hotel_bookings", {})]) for i in range(3)]
    script.append(LLMResponse(content="No priced hotels yet; try different dates."))
    ctx = AgentContext(org_id=ORG)
    result = await HotelSearchAgent().run(ctx, "loop", ScriptedLLM(script), max_iterations=3)
    assert len(ctx.tool_calls) == 3                      # capped at max_iterations
    assert result.output == "No priced hotels yet; try different dates."   # not a dead end


import pytest
from runtime import AgentContext, ToolCall
from agents.hotel_search_agent import (
    HotelSearchAgent, ROLE, GRANTED_MODULES, ALLOWED_TOOLS,
    MEM_SESSION_ID, MEM_OPTION_REF, MEM_CONFIRMED_PRICE, MEM_PARAMS,
)

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def _ctx():
    return AgentContext(org_id=ORG, currency="USD", nationality="AE")


def test_role_modules_and_no_write_tools():
    a = HotelSearchAgent()
    assert a.get_role() == ROLE == "hotel_search_agent"
    assert "hotels" in GRANTED_MODULES
    assert ALLOWED_TOOLS.isdisjoint({"book_hotel", "cancel_hotel", "send_hotel_cancel_request"})


def test_prompt_covers_the_spec():
    p = HotelSearchAgent().build_prompt(_ctx())
    assert ORG in p
    assert "never ask the user for it" in p.lower()
    assert "5 cheapest" in p and "ascending" in p
    assert "refresh_hotel_price" in p
    assert "1 room" in p and "2 adults" in p
    assert "refundable" in p and "meal plan" in p
    assert "never book" in p.lower()


def test_on_tool_result_populates_memory():
    a, ctx = HotelSearchAgent(), _ctx()
    a.on_tool_result(ctx, ToolCall("search_hotel_availability",
        {"city": "Metroville", "checkIn": "2026-08-15", "checkOut": "2026-08-20", "adults": 2},
        {"uuid": "U1", "hotels": []}))
    a.on_tool_result(ctx, ToolCall("refresh_hotel_price", {"optionRefId": "OPT-9"}, {"price": 340.0}))
    assert ctx.recall(MEM_SESSION_ID) == "U1"
    assert ctx.recall(MEM_OPTION_REF) == "OPT-9"
    assert ctx.recall(MEM_CONFIRMED_PRICE) == 340.0
    assert ctx.recall(MEM_PARAMS)["city"] == "Metroville"


@pytest.mark.asyncio
async def test_verify_clean_run():
    a, ctx = HotelSearchAgent(), _ctx()
    ctx.tool_calls = [
        ToolCall("search_hotel_availability", {"organizationId": ORG}, {"uuid": "U1"}),
        ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"}, {"price": 340}),
    ]
    for c in ctx.tool_calls:
        a.on_tool_result(ctx, c)
    res = await a.verify(ctx)
    assert res.passed, res.issues


@pytest.mark.asyncio
async def test_verify_flags_out_of_scope_tool():
    a, ctx = HotelSearchAgent(), _ctx()
    ctx.tool_calls = [ToolCall("book_hotel", {"organizationId": ORG}, {})]
    res = await a.verify(ctx)
    assert not res.passed and any("book_hotel" in i for i in res.issues)


@pytest.mark.asyncio
async def test_verify_flags_reprice_without_memory():
    a, ctx = HotelSearchAgent(), _ctx()
    ctx.tool_calls = [ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"}, {"price": 1})]
    res = await a.verify(ctx)
    assert not res.passed and any(MEM_OPTION_REF in i for i in res.issues)


@pytest.mark.asyncio
async def test_verify_flags_cross_tenant():
    a, ctx = HotelSearchAgent(), _ctx()
    ctx.tool_calls = [ToolCall("search_hotel_availability", {"organizationId": "OTHER"}, {"uuid": "U"})]
    res = await a.verify(ctx)
    assert not res.passed and any("expected" in i for i in res.issues)


# --------------------------------------------------------------------------
# The retrieval-first instruction is deliberately narrow: it covers enrichment
# questions and must not loosen how a hotel search handles a missing city.
# --------------------------------------------------------------------------

def _hotel_prompt():
    from runtime import AgentContext
    return HotelSearchAgent().build_prompt(AgentContext(org_id=ORG))


def test_enrichment_questions_search_before_asking_which_one():
    p = _hotel_prompt()
    assert "call search_enrichment first even if the message does not repeat the name" in p
    assert "ask which one they mean only if it returns no matches" in p


def test_a_hotel_search_is_classified_first_and_never_routed_to_enrichment():
    """The retrieval-first rule must not swallow an ordinary search. "a 4-star place
    with a pool and good reviews" mentions facilities and reputation and names no
    city — the prompt has to settle that as a search before the rule can apply."""
    p = _hotel_prompt()
    assert "Decide first whether the message is a request to find hotels" in p
    assert "a 4-star place with a pool and good reviews" in p
    assert "Do not call search_enrichment for it." in p
    assert "if no city is named, ask for one" in p
    # the rule is gated on the subject already being in play
    assert "Only when the message is a question about a hotel or city already in play" in p
    # and the original ambiguity rule is untouched
    assert "If the city name is ambiguous" in p


def test_the_agent_can_still_reach_search_enrichment():
    from agents.hotel_search_agent import ALLOWED_TOOLS
    assert "search_enrichment" in ALLOWED_TOOLS
    assert "book_hotel" not in ALLOWED_TOOLS and "cancel_hotel" not in ALLOWED_TOOLS
