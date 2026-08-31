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
    a.on_run_end(ctx, "Confirmed price: $340.00 USD total for the Alpha Hotel.")
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


# ---------------------------------------------------------------------------
# verify() reads the answer, not just the tool calls. Every fault below was
# produced by a running agent and passed with a green badge before these ran.
# ---------------------------------------------------------------------------
from agents.hotel_search_agent import MEM_ANSWER


async def _verify(answer, tool_calls, memory=None):
    a, ctx = HotelSearchAgent(), _ctx()
    ctx.tool_calls = list(tool_calls)
    for key, value in (memory or {}).items():
        ctx.remember(key, value)
    a.on_run_end(ctx, answer)
    return await a.verify(ctx)


def _enriched(payload=None):
    return ToolCall("enrich_destination", {"city": "Jeddah", "organizationId": ORG},
                    payload if payload is not None else {"city": "Jeddah", "domains": {}})


@pytest.mark.asyncio
async def test_on_run_end_keeps_the_answer_for_verify():
    a, ctx = HotelSearchAgent(), _ctx()
    a.on_run_end(ctx, "the answer")
    assert ctx.recall(MEM_ANSWER) == "the answer"


@pytest.mark.asyncio
async def test_estimated_weather_fails_verification():
    """Live: the stay was 1-4 September and Open-Meteo returned 10-13. The model
    noticed, then filled the gap with numbers that changed on each run."""
    answer = ("Based on typical September patterns in Jeddah, you can expect "
              "highs around 35°C during your stay.")
    result = await _verify(answer, [_enriched()])
    assert not result.passed
    assert any("estimates rather than reports" in i for i in result.issues)


@pytest.mark.asyncio
async def test_reporting_only_what_came_back_passes():
    """The fixture now carries the day it quotes, as a real run does — the
    measurement check reads the tool result, so an answer citing figures the
    result never held is flagged, which is the point of it."""
    answer = ("The forecast covers 10-13 September; 13 September is 28.5-35.1°C. "
              "It does not cover 1-4 September, so I have no data for your dates.")
    covered = _enriched({"city": "Jeddah", "domains": {"weather": {"findings": {
        "forecast_2026-09-13": [{"value": "28.5–35.1°C, 0 mm rain"}]}}}})
    result = await _verify(answer, [covered])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_estimation_check_does_not_fire_without_an_enrichment_tool():
    """A plain search may legitimately say check-in is typically 15:00."""
    answer = "Check-in is typically 15:00 and check-out 12:00."
    result = await _verify(answer, [ToolCall("search_hotel_availability",
                                             {"organizationId": ORG}, {"uuid": "U1"})])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_failed_reprice_reported_as_confirmed_fails():
    """Live, two lines apart: "there's a technical issue with the live rate
    confirmation tool", then "Confirmed Price: $150.92 USD total"."""
    a, ctx = HotelSearchAgent(), _ctx()
    call = ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"},
                    {"error": "HasuraGraphQLError: session expired"})
    a.on_tool_result(ctx, call)
    ctx.tool_calls = [call]
    a.on_run_end(ctx, "There was a technical issue with the tool. "
                      "**Confirmed Price:** $150.92 USD total.")
    result = await a.verify(ctx)
    assert not result.passed
    assert any("presents a rate as confirmed" in i for i in result.issues)


def test_a_failed_reprice_stores_no_price():
    a, ctx = HotelSearchAgent(), _ctx()
    a.on_tool_result(ctx, ToolCall("refresh_hotel_price", {"optionRefId": "OPT-9"},
                                   {"error": "supplier timeout"}))
    assert ctx.recall(MEM_CONFIRMED_PRICE) is None, "an error dict is not a price"


@pytest.mark.asyncio
async def test_failed_reprice_reported_honestly_passes():
    a, ctx = HotelSearchAgent(), _ctx()
    call = ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"},
                    {"error": "supplier timeout"})
    a.on_tool_result(ctx, call)
    # The price it reports came from the search earlier in the session, which is
    # still in ctx.tool_calls — that is what makes reporting it honest.
    ctx.tool_calls = [ToolCall("get_hotel_options", {"organizationId": ORG},
                               {"options": [{"price": {"totalPrice": 150.92,
                                                       "currency": "USD"}}]}), call]
    a.on_run_end(ctx, "The live rate could not be confirmed. The last price I saw "
                      "was $150.92, which is not confirmed.")
    result = await a.verify(ctx)
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_successful_reprice_still_stores_and_passes():
    a, ctx = HotelSearchAgent(), _ctx()
    call = ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"},
                    {"price": 340.0})
    a.on_tool_result(ctx, call)
    ctx.tool_calls = [call]
    a.on_run_end(ctx, "Confirmed price: $340.00 USD total.")
    result = await a.verify(ctx)
    assert ctx.recall(MEM_CONFIRMED_PRICE) == 340.0
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_option_reference_in_the_answer_fails():
    answer = ("Your room: KING BED ROOM, option "
              "33!~|a0!~|b260910!~|c260913!~|d5654415!~|e0!~| — $124.26 total.")
    result = await _verify(answer, [ToolCall("get_hotel_options", {"organizationId": ORG}, {})])
    assert not result.passed
    assert any("supplier option reference" in i for i in result.issues)


@pytest.mark.asyncio
async def test_promised_memory_without_the_tool_fails():
    """Live: "I'll remember that preference for future bookings." with no tool
    chip at all, and the Memory panel still holding the old value."""
    result = await _verify("I'll remember that preference for future bookings.",
                           [ToolCall("search_hotel_availability", {"organizationId": ORG}, {})])
    assert not result.passed
    assert any("remember_preference was never called" in i for i in result.issues)


@pytest.mark.asyncio
async def test_promised_memory_with_the_tool_passes():
    result = await _verify("I've noted your preference for 4-star hotels.",
                           [ToolCall("remember_preference",
                                     {"statement": "4-star or better", "key": "hotel_stars"},
                                     {"stored": True})])
    assert result.passed, result.issues


def _status_transport(script):
    """Replies in order; each entry is (status, body)."""
    seen = {"n": 0, "sent": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["sent"].append(json.loads(request.content.decode()))
        status, body = script[min(seen["n"], len(script) - 1)]
        seen["n"] += 1
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), seen


_AFFORD = {"error": {"message": "This request requires more credits, or fewer "
                                "max_tokens. You requested up to 4000 tokens, but "
                                "can only afford 46.", "code": 402}}


@pytest.mark.asyncio
async def test_402_retry_uses_the_budget_openrouter_named():
    """The old floor of 256 guaranteed a second 402 whenever the affordable
    budget was below it, so the documented self-heal never ran on a nearly
    drained key — and the page called it "no credit left"."""
    transport, seen = _status_transport([
        (402, _AFFORD),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="z-ai/glm-5.2", max_tokens=4000,
                        transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert seen["sent"][0]["max_tokens"] == 4000
    assert seen["sent"][1]["max_tokens"] == 38, "must fit inside the 46 it named"


@pytest.mark.asyncio
async def test_a_naming_404_retries_the_other_spelling():
    """z-ai/glm-5.2:free is not published; z-ai/glm-5.2 is."""
    transport, seen = _status_transport([
        (404, {"error": {"message": "no such model", "code": 404}}),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="z-ai/glm-5.2:free", transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert [m["model"] for m in seen["sent"]] == ["z-ai/glm-5.2:free", "z-ai/glm-5.2"]
    assert llm.model == "z-ai/glm-5.2"


@pytest.mark.asyncio
async def test_the_spelling_flip_happens_only_once():
    transport, seen = _status_transport([(404, {"error": {"message": "no such model"}})])
    llm = OpenRouterLLM(api_key="k", model="a/b:free", transport=transport)
    with pytest.raises(LLMError):
        await llm.complete([{"role": "user", "content": "hi"}])
    assert len(seen["sent"]) == 2, "one flip, then it gives up rather than looping"


@pytest.mark.asyncio
async def test_a_spelling_flip_does_not_consume_the_budget_retry():
    """Gating the budget retry on attempt==0 meant a spelling flip on the first
    attempt spent its only chance, so the 402 that followed was raised with the
    original 4000 still in it and no retry ever happened."""
    transport, seen = _status_transport([
        (404, {"error": {"message": "no such model"}}),
        (402, _AFFORD),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="poolside/laguna-s-2.1:free",
                        max_tokens=4000, transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert [m["model"] for m in seen["sent"]] == [
        "poolside/laguna-s-2.1:free", "poolside/laguna-s-2.1", "poolside/laguna-s-2.1"]
    assert [m["max_tokens"] for m in seen["sent"]] == [4000, 4000, 38]


@pytest.mark.asyncio
async def test_no_endpoints_found_also_retries_the_other_spelling():
    """inclusionai/ling-3.0-flash-fin is published only as :free. The 404 for
    the plain name reads "No endpoints found for <id>." and never says "model",
    so the guard that looked for that word let it through untouched."""
    transport, seen = _status_transport([
        (404, {"error": {"message": "No endpoints found for inclusionai/ling-3.0-flash-fin."}}),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="inclusionai/ling-3.0-flash-fin",
                        transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert llm.model == "inclusionai/ling-3.0-flash-fin:free"


@pytest.mark.asyncio
async def test_the_free_flip_is_openrouter_only():
    """:free is an OpenRouter naming convention. Appending it on Groq turned
    "tool calling is not supported with this model" into "the model
    `groq/compound:free` does not exist" — a different, misleading error."""
    transport, seen = _status_transport([
        (400, {"error": {"message": "`tool calling` is not supported with this model"}}),
    ])
    llm = OpenRouterLLM(api_key="gsk_x", model="groq/compound",
                        base_url="https://api.groq.com/openai/v1", transport=transport)
    with pytest.raises(LLMError) as exc:
        await llm.complete([{"role": "user", "content": "hi"}])
    assert "tool calling" in str(exc.value)
    assert len(seen["sent"]) == 1, "no flip off OpenRouter"
    assert llm.model == "groq/compound"


# ---------------------------------------------------------------------------
# Stopping is not answering. Models that reason in a separate field return
# content="" and put everything there; the turn then ended on "(no reply)" with
# the tools already called and the badge still green.
# ---------------------------------------------------------------------------

class _Scripted:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete(self, messages, tools=None):
        self.calls.append({"tools_offered": len(tools or []),
                           "last_role": messages[-1]["role"]})
        return self._responses.pop(0)

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_an_empty_final_message_is_not_accepted_as_the_answer():
    from runtime import LLMResponse
    llm = _Scripted([
        LLMResponse(content=""),                      # stops, writes nothing
        LLMResponse(content="Here are the 5 cheapest hotels."),   # forced answer
    ])
    ctx = _ctx()
    res = await HotelSearchAgent().run(ctx, "find hotels", llm, max_iterations=4)
    assert res.output == "Here are the 5 cheapest hotels."
    assert len(llm.calls) == 2
    assert llm.calls[1]["tools_offered"] == 0, "the forced answer must not offer tools"
    assert [m["role"] for m in res.messages][-2:] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_a_whitespace_only_answer_counts_as_empty():
    from runtime import LLMResponse
    llm = _Scripted([LLMResponse(content="   \n  "), LLMResponse(content="Real answer.")])
    res = await HotelSearchAgent().run(_ctx(), "find hotels", llm, max_iterations=4)
    assert res.output == "Real answer."


@pytest.mark.asyncio
async def test_an_empty_answer_fails_verification():
    result = await _verify("", [ToolCall("search_hotel_availability", {"organizationId": ORG}, {})])
    assert not result.passed
    assert any("no written answer" in i for i in result.issues)


@pytest.mark.asyncio
async def test_an_empty_triage_report_fails_verification():
    from agents.ops_triage_agent import OpsTriageAgent
    a, ctx = OpsTriageAgent(), AgentContext(org_id=ORG)
    a.on_run_start(ctx)
    a.on_run_end(ctx, "")
    result = await a.verify(ctx)
    assert not result.passed
    assert any("no written report" in i for i in result.issues)


@pytest.mark.asyncio
async def test_a_flip_that_lands_nowhere_reports_the_original_failure():
    """A naming 404 on one spelling flipped to the other, which also 404s — so
    the failure was reported against a name nobody configured."""
    transport, seen = _status_transport([
        (404, {"error": {"message": "no such model", "code": 404}}),
        (404, {"error": {"message": "No endpoints found for inclusionai/ling-3.0-flash-fin."}}),
    ])
    llm = OpenRouterLLM(api_key="k", model="inclusionai/ling-3.0-flash-fin:free",
                        transport=transport)
    with pytest.raises(LLMError) as exc:
        await llm.complete([{"role": "user", "content": "hi"}])
    message = str(exc.value)
    assert "404" in message and "no such model" in message
    assert "inclusionai/ling-3.0-flash-fin:free" in message, "and it names the configured model"
    assert "No endpoints found" not in message
    assert llm.model == "inclusionai/ling-3.0-flash-fin:free", "the name is restored"


@pytest.mark.asyncio
async def test_provider_tool_call_fields_survive_the_round_trip():
    """Gemini attaches a thought_signature under extra_content and rejects the
    next turn without it: "Function call is missing a thought_signature in
    functionCall parts". Rebuilding the call from name+arguments dropped it."""
    from runtime import LLMResponse
    signed = {"id": "c1", "type": "function",
              "extra_content": {"google": {"thought_signature": "EukCCuYCARFNMg"}},
              "function": {"name": "search_hotel_availability", "arguments": "{}"}}
    transport, seen = _status_transport([
        (200, {"choices": [{"message": {"role": "assistant", "tool_calls": [signed]}}]}),
        (200, {"choices": [{"message": {"content": "done"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="gemini-flash-latest",
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                        transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.tool_calls[0].raw == signed, "the provider's dict is kept verbatim"

    ctx = _ctx()
    llm2 = OpenRouterLLM(api_key="k", model="gemini-flash-latest",
                         base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                         transport=_status_transport([
                             (200, {"choices": [{"message": {"role": "assistant",
                                                             "tool_calls": [signed]}}]}),
                             (200, {"choices": [{"message": {"content": "done"}}]}),
                         ])[0])
    res = await HotelSearchAgent().run(ctx, "find hotels", llm2, max_iterations=3)
    echoed = next(m for m in res.messages if m.get("tool_calls"))
    assert echoed["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "EukCCuYCARFNMg"


@pytest.mark.asyncio
async def test_a_503_is_retried_not_surfaced_as_a_dead_end():
    """Google answers 503 "experiencing high demand" and clears on retry; the
    client stopped after one attempt and reported it as final."""
    import runtime
    transport, seen = _status_transport([
        (503, {"error": {"code": 503, "message": "This model is currently experiencing high demand."}}),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    runtime._BACKOFF = (0.0, 0.0)
    llm = OpenRouterLLM(api_key="k", model="gemini-flash-latest",
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                        transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert len(seen["sent"]) == 2


@pytest.mark.asyncio
async def test_errors_name_the_host_that_answered():
    """Four providers share this client; "OpenRouter HTTP 503" was wrong for
    three of them and sent people to the wrong dashboard."""
    transport, _ = _status_transport([(400, {"error": {"message": "bad request"}})])
    llm = OpenRouterLLM(api_key="k", model="gpt-4o-mini",
                        base_url="https://api.openai.com/v1", transport=transport)
    with pytest.raises(LLMError) as exc:
        await llm.complete([{"role": "user", "content": "hi"}])
    assert "api.openai.com" in str(exc.value)
    assert "OpenRouter" not in str(exc.value)


@pytest.mark.asyncio
async def test_a_rate_limit_no_longer_flips_a_free_name_to_its_paid_twin():
    """429 is a capacity problem, not a naming one. Flipping :free off could
    land on the paid model; it is retried on the same name instead."""
    import runtime
    runtime._BACKOFF = (0.0, 0.0)
    transport, seen = _status_transport([
        (429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}}),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="inclusionai/ling-3.0-flash-fin:free",
                        transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert [m["model"] for m in seen["sent"]] == [
        "inclusionai/ling-3.0-flash-fin:free"] * 2, "same name both times"
    assert llm.model == "inclusionai/ling-3.0-flash-fin:free"


# ---------------------------------------------------------------------------
# verify() read the wording and the tool names but never compared them, so a
# search returning "Real Hotel" and an answer saying "Fiction Hotel costs $999"
# passed with zero issues.
# ---------------------------------------------------------------------------

def _search_call(name="Real Hotel", total=125.51):
    return ToolCall("search_hotel_availability", {"organizationId": ORG, "city": "Jeddah"},
                    {"uuid": "U1", "nights": 3,
                     "hotels": [{"hotelName": name, "hotelCode": "1442211",
                                 "pricePerNight": 41.84,
                                 "price": {"totalPrice": total, "currency": "USD"}}]})


@pytest.mark.asyncio
async def test_a_price_no_tool_returned_fails_verification():
    result = await _verify("Fiction Hotel costs $999 USD total for your stay.",
                           [_search_call()])
    assert not result.passed
    assert any("quotes prices no tool returned" in i for i in result.issues)


@pytest.mark.asyncio
async def test_prices_the_tool_did_return_pass():
    result = await _verify(
        "Real Hotel — $41.84 per night, $125.51 total for 3 nights.", [_search_call()])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_rounded_and_comma_spellings_of_a_real_price_pass():
    result = await _verify("Real Hotel is about $126 total.", [_search_call()])
    assert result.passed, result.issues
    big = await _verify("Suite is $1,250.00 total.", [_search_call(total=1250.0)])
    assert big.passed, big.issues


@pytest.mark.asyncio
async def test_the_price_check_is_silent_without_a_priced_tool():
    """An enrichment-only turn quotes no supplier prices and must not be graded
    against a search that never ran."""
    result = await _verify("It will be hot and dry.", [_enriched()])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_carrying_one_windows_numbers_to_another_fails_verification():
    """Live: a forecast covering 10-13 September, answered with "Expect
    similarly hot, dry weather for your Sep 1-2 dates" — verified green."""
    result = await _verify(
        "The forecast covers Sep 10-13. Expect similarly hot, dry weather for "
        "your Sep 1-2 dates.", [_enriched()])
    assert not result.passed
    assert any("estimates rather than reports" in i for i in result.issues)


@pytest.mark.asyncio
async def test_reporting_the_window_it_actually_covers_passes():
    result = await _verify(
        "The forecast covers Sep 10-13 at 28.5-35.1 C with no rain. It does not "
        "cover Sep 1-2, so I have no data for your dates.", [_enriched()])
    assert result.passed, result.issues


def test_the_prompt_refuses_to_invent_dates():
    """"Next month" was answered with a search for Sep 1 to Sep 2 — a one-night
    stay nobody asked for, which made every price in that demo wrong."""
    prompt = HotelSearchAgent().build_prompt(_ctx())
    assert "Dates do not" in prompt
    assert "do not assume one night" in prompt


@pytest.mark.asyncio
async def test_an_honestly_reported_failed_reprice_is_not_flagged():
    """Live: "the supplier returned an error and could not provide a confirmed
    price" failed verification, because it contains "confirmed price". The
    honest answer was the one that got the red badge."""
    a, ctx = HotelSearchAgent(), _ctx()
    call = ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"},
                    {"error": "supplier returned no price"})
    a.on_tool_result(ctx, call)
    ctx.tool_calls = [_search_call(total=125.78), call]
    a.on_run_end(ctx, "I tried to confirm the live rate for Loren Suites, but the "
                      "supplier's system returned an error and could not provide a "
                      "confirmed price at this moment. Last seen price (not "
                      "confirmed): $125.78 total.")
    result = await a.verify(ctx)
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_a_rate_actually_claimed_as_confirmed_still_fails():
    a, ctx = HotelSearchAgent(), _ctx()
    call = ToolCall("refresh_hotel_price", {"organizationId": ORG, "optionRefId": "OPT-9"},
                    {"error": "supplier returned no price"})
    a.on_tool_result(ctx, call)
    ctx.tool_calls = [_search_call(total=125.78), call]
    a.on_run_end(ctx, "There was a technical issue with the tool. "
                      "Confirmed Price: $125.78 total.")
    result = await a.verify(ctx)
    assert not result.passed
    assert any("presents a rate as confirmed" in i for i in result.issues)


@pytest.mark.asyncio
async def test_malformed_tool_arguments_do_not_end_the_turn():
    """Live: "tool call arguments were not valid JSON: Expecting ',' delimiter"
    killed the conversation outright, with the earlier tools already run."""
    transport, seen = _status_transport([
        (200, {"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "search_hotel_availability",
                         "arguments": '{"city": "Jeddah" "checkIn": "2026-09-01"}'}}]}}]}),
        (200, {"choices": [{"message": {"content": "Recovered and answered."}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="m", transport=transport)
    first = await llm.complete([{"role": "user", "content": "hi"}])
    assert first.tool_calls[0].invalid_arguments, "the parse failure is carried, not raised"
    assert first.tool_calls[0].arguments == {}

    ctx = _ctx()
    llm2 = OpenRouterLLM(api_key="k", model="m", transport=_status_transport([
        (200, {"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "search_hotel_availability",
                         "arguments": '{"city": "Jeddah" "bad"}'}}]}}]}),
        (200, {"choices": [{"message": {"content": "Recovered and answered."}}]}),
    ])[0])
    res = await HotelSearchAgent().run(ctx, "find hotels", llm2, max_iterations=4)
    assert res.output == "Recovered and answered."
    assert ctx.tool_calls == [], "a call that never ran is not recorded as one"
    told = next(m for m in res.messages if m.get("role") == "tool")
    assert "not valid JSON" in told["content"]


# ---------------------------------------------------------------------------
# Context growth. Measured on this agent before the caps: each turn added ~640
# prompt tokens, so turn 10 cost 8,333 against turn 1's 2,564 — before the
# 2,228 tokens of tool specs. Cost, latency and every prompt ceiling scale with
# that, which is why later turns failed where earlier ones had not.
# ---------------------------------------------------------------------------
from runtime import (MAX_HISTORY_MESSAGES, MAX_TOOL_RESULT_CHARS, cap_tool_result,
                     trim_history)


def _turn(n):
    return [
        {"role": "user", "content": f"question {n}"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{n}", "type": "function",
             "function": {"name": "search_hotel_availability", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{n}", "content": "{}"},
        {"role": "assistant", "content": f"answer {n}"},
    ]


def test_history_is_bounded_and_keeps_the_system_prompt():
    messages = [{"role": "system", "content": "S"}]
    for n in range(20):
        messages.extend(_turn(n))
        messages = trim_history(messages)
    assert messages[0]["role"] == "system"
    assert len(messages) <= MAX_HISTORY_MESSAGES + 1
    assert messages[-1]["content"] == "answer 19", "the newest turn survives"


def test_the_cut_never_orphans_a_tool_result():
    """A `tool` message whose assistant tool_calls were dropped is rejected
    outright by an OpenAI-compatible endpoint — a shorter transcript that no
    longer works. The window may only open on a user message."""
    messages = [{"role": "system", "content": "S"}]
    for n in range(30):
        messages.extend(_turn(n))
        messages = trim_history(messages)
        body = messages[1:]
        assert not body or body[0]["role"] == "user", body[0]["role"]
        answered = set()
        for m in body:
            for tc in m.get("tool_calls") or []:
                answered.add(tc["id"])
            if m["role"] == "tool":
                assert m["tool_call_id"] in answered, "orphaned tool result"


def test_a_short_conversation_is_left_alone():
    messages = [{"role": "system", "content": "S"}, *_turn(1)]
    assert trim_history(messages) == messages


def test_one_very_long_exchange_is_kept_whole_rather_than_broken():
    """No user boundary ahead of the cut means any cut would orphan something."""
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    for n in range(40):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{n}", "content": "{}"})
    assert trim_history(messages) == messages


def test_a_large_tool_result_is_capped_for_the_model_only():
    payload = "x" * (MAX_TOOL_RESULT_CHARS + 5000)
    capped = cap_tool_result(payload)
    assert len(capped) < len(payload)
    assert "truncated 5000 characters" in capped
    assert "pageNumber" in capped, "it says how to avoid the trim"
    assert cap_tool_result("small") == "small"


@pytest.mark.asyncio
async def test_verify_still_sees_the_full_result_after_capping():
    """The cap is on the transcript, not on ctx.tool_calls — the price
    cross-check reads the recorded result and must stay exact."""
    from runtime import LLMResponse
    big = {"uuid": "U1", "hotels": [{"hotelName": f"H{i}", "price": {"totalPrice": 100.0 + i}}
                                    for i in range(400)]}
    llm = _Scripted([
        LLMResponse(tool_calls=[LLMToolCall("c1", "search_hotel_availability", {})]),
        LLMResponse(content="The cheapest is H7 at $107.00 total."),
    ])
    ctx = _ctx()

    import runtime
    async def fake_dispatch(name, args, context):
        return big
    original = runtime.dispatch
    runtime.dispatch = fake_dispatch
    try:
        res = await HotelSearchAgent().run(ctx, "find hotels", llm, max_iterations=4)
    finally:
        runtime.dispatch = original

    tool_message = next(m for m in res.messages if m["role"] == "tool")
    assert len(tool_message["content"]) <= MAX_TOOL_RESULT_CHARS + 200, "capped in the transcript"
    assert len(ctx.tool_calls[0].result["hotels"]) == 400, "whole result recorded"
    assert res.verification.passed, res.verification.issues


@pytest.mark.asyncio
async def test_a_per_night_price_derived_from_a_returned_total_passes():
    """Live false positive: "$41.93, $41.97, $44.16" flagged as invented. They
    are returned totals divided by the returned nights, which is arithmetic on
    the tool's own numbers — and the only option once paging returns no
    pricePerNight."""
    search = ToolCall("search_hotel_availability", {"organizationId": ORG}, {
        "uuid": "U1", "nights": 3, "hotels": [
            {"hotelName": "Loren Suites", "pricePerNight": 41.93,
             "price": {"totalPrice": 125.78, "currency": "USD"}}]})
    paging = ToolCall("get_hotel_search_results", {"organizationId": ORG}, {
        "isComplete": True, "hotels": [
            {"hotelName": "Carawan", "pricePerNight": None,
             "price": {"totalPrice": 132.47, "currency": "USD"}}]})
    result = await _verify(
        "Loren Suites $41.93/night, $125.78 total. Carawan $44.16/night, "
        "$132.47 total.", [search, paging])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_division_does_not_excuse_an_invented_price():
    search = ToolCall("search_hotel_availability", {"organizationId": ORG},
                      {"uuid": "U1", "nights": 3, "hotels": [
                          {"hotelName": "Loren Suites",
                           "price": {"totalPrice": 125.78, "currency": "USD"}}]})
    result = await _verify("Fiction Hotel is $999 total.", [search])
    assert not result.passed
    assert any("quotes prices no tool returned" in i for i in result.issues)


@pytest.mark.asyncio
async def test_a_cloudflare_524_is_retried():
    """Several of these providers sit behind Cloudflare. api.llm7.io answered a
    second turn with 524 "a timeout occurred" — the same transient class as a
    504, and it was being raised as final."""
    import runtime
    runtime._BACKOFF = (0.0, 0.0)
    transport, seen = _status_transport([
        (524, {"error": {"message": "a timeout occurred"}}),
        (200, {"choices": [{"message": {"content": "ok"}}]}),
    ])
    llm = OpenRouterLLM(api_key="k", model="minimax-m2.7",
                        base_url="https://api.llm7.io/v1", transport=transport)
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert len(seen["sent"]) == 2


def test_every_cloudflare_origin_error_is_transient():
    from runtime import _TRANSIENT
    for status in (520, 521, 522, 523, 524):
        assert status in _TRANSIENT, status
    assert 400 not in _TRANSIENT and 402 not in _TRANSIENT and 404 not in _TRANSIENT


# ---------------------------------------------------------------------------
# Reported live on Makkah 4-8 September: retrieval worked, the write-up did not.
# Day 4 carried day 5's figures and each day behind it shifted. Every number was
# real, so set membership could not see it — and "0 mm rain" is 0 on every dry
# day, which masked the temperatures when the test was "any figure matches".
# ---------------------------------------------------------------------------

_MAKKAH = {"city": "Makkah", "domains": {"weather": {"findings": {
    "forecast_2026-09-04": [{"value": "31.3–40.7°C, 0 mm rain"}],
    "forecast_2026-09-05": [{"value": "28.4–40.4°C, 0 mm rain"}],
    "forecast_2026-09-06": [{"value": "27.2–39.8°C, 0 mm rain"}],
    "forecast_2026-09-07": [{"value": "30.4–38.9°C, 0 mm rain"}],
    "forecast_2026-09-08": [{"value": "29.9–40.4°C, 0 mm rain"}]}}}}


def _makkah_call():
    return ToolCall("enrich_destination", {"city": "Makkah", "organizationId": ORG}, _MAKKAH)


@pytest.mark.asyncio
async def test_a_day_given_another_days_figures_is_flagged():
    drifted = ("4 Sep: 28–40 °C, 0 mm rain\n5 Sep: 27–40 °C, 0 mm rain\n"
               "6 Sep: 27–40 °C, 0 mm rain\n7 Sep: 30–39 °C, 0 mm rain\n"
               "8 Sep: 30–40 °C, 0 mm rain")
    result = await _verify(drifted, [_makkah_call()])
    assert not result.passed
    issues = " ".join(result.issues)
    assert "another day's figures" in issues
    assert "09-04" in issues and "09-05" in issues


@pytest.mark.asyncio
async def test_the_same_series_stated_faithfully_passes():
    faithful = ("4 Sep: 31.3–40.7 °C, 0 mm rain\n5 Sep: 28.4–40.4 °C, 0 mm rain\n"
                "6 Sep: 27.2–39.8 °C, 0 mm rain\n7 Sep: 30.4–38.9 °C, 0 mm rain\n"
                "8 Sep: 29.9–40.4 °C, 0 mm rain")
    result = await _verify(faithful, [_makkah_call()])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_rounding_to_the_right_day_is_still_allowed():
    """31.3 stated as 31 is reporting; 31.3 stated as 28.4's value is not."""
    rounded = ("4 Sep: 31–41 °C, 0 mm rain\n5 Sep: 28–40 °C, 0 mm rain")
    result = await _verify(rounded, [_makkah_call()])
    assert result.passed, result.issues


@pytest.mark.asyncio
async def test_an_iso_dated_line_is_checked_too():
    result = await _verify("2026-09-04: 28–40 °C", [_makkah_call()])
    assert not result.passed
    assert "09-04" in " ".join(result.issues)


@pytest.mark.asyncio
async def test_prose_with_no_day_reference_is_left_alone():
    """Not every enrichment answer is a table; a summary naming no day cannot be
    misaligned and must not be graded as if it were."""
    result = await _verify("Hot and dry throughout, with no rain expected.",
                           [_makkah_call()])
    assert result.passed, result.issues


def test_a_range_yields_both_of_its_numbers():
    from agents.hotel_search_agent import _quoted_measures
    assert set(_quoted_measures("28–40 °C, 0 mm rain")) == {"28", "40", "0"}
    assert set(_quoted_measures("31.3–40.7 °C")) == {"31.3", "40.7"}
