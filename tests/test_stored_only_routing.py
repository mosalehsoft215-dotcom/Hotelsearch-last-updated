"""When the user says not to fetch, no tool goes out to the web.

search_enrichment stays open — it only reads what an earlier fetch already
stored, which is exactly what "use stored enrichment only" asks for. Everything
else is refused before it runs, so the refusal is a property of the tool calls
the run actually made, not of what the model was asked to do.
"""
import pytest

import web_tools
from agents.hotel_search_agent import HotelSearchAgent
from enrichment_index import EnrichmentIndex, SqliteVectorStore
from runtime import (
    ENRICHMENT_FETCH_TOOLS, AgentContext, LLMResponse, LLMToolCall, StoredOnlyRefusal,
    dispatch, is_stored_only,
)
from web_enrich import Cache, Claim, Enricher, Enrichment, Source

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"

PHRASES = [
    "For Riyadh, using stored enrichment only, what is the weather on 4 September 2026?",
    "What is the weather in Riyadh on 4 September 2026? Do not fetch.",
    "Weather in Riyadh 4 September 2026 — do not use fresh data.",
    "Use existing enrichment only: what is the weather in Riyadh?",
    "Tell me about Riyadh without fetching anything new.",
    "Use only the stored enrichment for Riyadh.",
]

# Deliberately not in the list above. "What does the cached data say" reads as a
# question about the cache as easily as an instruction not to leave it, and this
# detector gates a hard refusal — it holds only phrasings that can mean one
# thing. The prompt covers the rest.
AMBIGUOUS = ["What does the cached data say about Riyadh?",
             "Is that from the index or from the web?"]


class Scripted:
    """Replies in order. Records the tool results the loop handed back, which is
    where a refusal shows up from the model's side."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.turns = []

    async def complete(self, messages, tools=None):
        self.turns.append(list(messages))
        return self.replies.pop(0) if self.replies else LLMResponse(content="done")

    def tool_messages(self):
        return [m for turn in self.turns for m in turn if m.get("role") == "tool"]


def call(name, **args):
    return LLMToolCall(id=f"c-{name}", name=name, arguments=args)


@pytest.fixture
def stored_index(monkeypatch):
    """An index holding one city's weather and nothing else, so "the entity is
    not in stored enrichment" can be told apart from "the index is empty"."""
    index = EnrichmentIndex(SqliteVectorStore(":memory:"))
    enrichment = Enrichment(subject="Riyadh", domain="weather",
                            entity_type="city", entity_ref="Riyadh")
    source = Source(url="https://api.open-meteo.com/v1/forecast?x=1",
                    title="Open-Meteo forecast", tier="official")
    enrichment.claims = [
        Claim(domain="weather", field_name="place", value="Riyadh, Saudi Arabia",
              sources=[source]),
        Claim(domain="weather", field_name="forecast_2026-09-04",
              value="28.4–40.4°C, 0 mm rain", sources=[source]),
    ]
    index.add(enrichment)
    monkeypatch.setattr(web_tools, "_index", index)
    monkeypatch.setattr(web_tools, "_enricher", Enricher([], Cache(), index=index))
    return index


# ---- recognising the instruction ----

@pytest.mark.parametrize("message", PHRASES)
def test_the_instruction_is_recognised(message):
    assert is_stored_only(message) is True


@pytest.mark.parametrize("message", [
    "What is the weather in Riyadh from 4 to 8 September 2026?",
    "Find me a 4-star hotel in Riyadh with a pool.",
    "Fetch the latest news for Riyadh.",
    "Do not book anything, just show me the options.",
    "Refresh the price for that room.",
])
def test_an_ordinary_request_is_not_restricted(message):
    assert is_stored_only(message) is False


@pytest.mark.parametrize("message", AMBIGUOUS)
def test_a_phrase_that_could_mean_two_things_does_not_trigger_a_refusal(message):
    assert is_stored_only(message) is False


# ---- the fetch tools do not run ----

@pytest.mark.asyncio
@pytest.mark.parametrize("message", PHRASES)
async def test_no_enrichment_fetch_tool_runs_in_any_no_fetch_case(message, stored_index):
    """The model is scripted to try a fetch on every one of these. None lands."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[call("enrich_destination", city="Riyadh",
                                     checkIn="2026-09-04", checkOut="2026-09-05")]),
        # What the agent actually asks the index — its own words, carrying the
        # subject and the date, not the user's phrasing verbatim.
        LLMResponse(tool_calls=[call("search_enrichment",
                                     question="weather in Riyadh on 4 September 2026")]),
        LLMResponse(content="Riyadh on 4 September 2026: 28.4–40.4°C, 0 mm rain, "
                            "from stored enrichment."),
    )
    result = await agent.run(ctx, message, llm, max_iterations=4)

    called = {c.name for c in ctx.tool_calls}
    assert called & ENRICHMENT_FETCH_TOOLS == set()
    assert called == {"search_enrichment"}
    assert result.verification.passed, result.verification.issues


@pytest.mark.asyncio
async def test_every_fetch_tool_is_refused_not_only_the_destination_one(stored_index):
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[
            call("enrich_destination", city="Riyadh"),
            call("enrich_hotel_info", hotelName="Carawan Al Fahad"),
            call("enrich_company_facts", companyName="Hilton Worldwide"),
            call("enrich_agency_facts", agencyName="Rihla Travel"),
        ]),
        LLMResponse(content="Nothing further is available in stored enrichment."),
    )
    await agent.run(ctx, "Do not fetch. What do we already hold?", llm, max_iterations=3)

    assert ctx.tool_calls == []
    refusals = llm.tool_messages()
    assert len(refusals) == 4
    for message in refusals:
        assert "was not run" in message["content"]
        assert "search_enrichment" in message["content"]


@pytest.mark.asyncio
async def test_the_refusal_tells_the_model_what_to_do_instead(stored_index):
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[call("enrich_destination", city="Riyadh")]),
        LLMResponse(content="Not available in stored enrichment."),
    )
    await agent.run(ctx, "Do not fetch. Weather in Riyadh?", llm, max_iterations=3)

    body = llm.tool_messages()[0]["content"]
    assert "stored enrichment only" in body
    assert "not available in stored enrichment" in body
    assert "do not fetch" in body


# ---- the three shapes of "it is not there" ----

@pytest.mark.asyncio
async def test_known_entity_unsupported_field_stays_on_search_enrichment(stored_index):
    """Riyadh is in the index; its air quality is not. The answer is that the
    field is unavailable, and no fetch is attempted to go and get it."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    message = ("Using stored enrichment only, what is the air quality index in "
               "Riyadh on 4 September 2026?")
    llm = Scripted(
        LLMResponse(tool_calls=[call("search_enrichment", question=message)]),
        LLMResponse(tool_calls=[call("enrich_destination", city="Riyadh")]),
        LLMResponse(content="Air quality is not available in stored enrichment for "
                            "Riyadh. Stored enrichment holds the forecast only."),
    )
    result = await agent.run(ctx, message, llm, max_iterations=4)

    assert {c.name for c in ctx.tool_calls} == {"search_enrichment"}
    assert result.verification.passed, result.verification.issues


@pytest.mark.asyncio
async def test_unknown_entity_stays_on_search_enrichment(stored_index):
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    message = "Using stored enrichment only, what is the weather in Aswan?"
    llm = Scripted(
        LLMResponse(tool_calls=[call("search_enrichment", question=message)]),
        LLMResponse(tool_calls=[call("enrich_destination", city="Aswan")]),
        LLMResponse(content="Aswan is not available in stored enrichment."),
    )
    result = await agent.run(ctx, message, llm, max_iterations=4)

    assert {c.name for c in ctx.tool_calls} == {"search_enrichment"}
    # It really did come back empty — the entity guard, not an accident of score.
    assert ctx.tool_calls[0].result["matches"] == []
    assert result.verification.passed, result.verification.issues


@pytest.mark.asyncio
async def test_a_date_the_index_does_not_hold_stays_on_search_enrichment(stored_index):
    """The index holds 4 September for Riyadh and no other day."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    message = ("Using stored enrichment only, what is the weather in Riyadh on "
               "19 December 2026?")
    llm = Scripted(
        LLMResponse(tool_calls=[call("search_enrichment", question=message)]),
        LLMResponse(tool_calls=[call("enrich_destination", city="Riyadh",
                                     checkIn="2026-12-19")]),
        LLMResponse(content="Stored enrichment holds 4 September 2026 for Riyadh. "
                            "19 December 2026 is not available in stored enrichment."),
    )
    result = await agent.run(ctx, message, llm, max_iterations=4)

    assert {c.name for c in ctx.tool_calls} == {"search_enrichment"}
    days = {m["field"] for m in ctx.tool_calls[0].result["matches"]}
    assert "forecast_2026-12-19" not in days
    assert result.verification.passed, result.verification.issues


# ---- the restriction is per turn, and travels with delegated work ----

@pytest.mark.asyncio
async def test_the_restriction_lifts_on_the_next_turn(stored_index):
    """A chat session reuses one context. "Do not fetch" on one turn must not
    still be blocking on the next, or the session quietly stops working."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)

    await agent.run(ctx, "Do not fetch. Weather in Riyadh?",
                    Scripted(LLMResponse(content="From stored enrichment only.")),
                    max_iterations=2)
    assert ctx.stored_only is True

    await agent.run(ctx, "Now what is the weather in Riyadh from 4 to 8 September?",
                    Scripted(LLMResponse(content="Let me look.")), max_iterations=2)
    assert ctx.stored_only is False


def test_a_delegated_child_keeps_the_restriction():
    parent = AgentContext(org_id=ORG)
    parent.stored_only = True
    assert parent.for_child("Check the weather in Riyadh.").stored_only is True

    unrestricted = AgentContext(org_id=ORG)
    assert unrestricted.for_child("Check the weather in Riyadh.").stored_only is False


# ---- the backstop ----

@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(ENRICHMENT_FETCH_TOOLS))
async def test_dispatch_refuses_a_fetch_even_when_called_directly(name, stored_index):
    """The run loop refuses first. This is the guard for anything that reaches
    dispatch another way."""
    ctx = AgentContext(org_id=ORG)
    ctx.stored_only = True
    with pytest.raises(StoredOnlyRefusal):
        await dispatch(name, {"city": "Riyadh", "hotelName": "X", "companyName": "X",
                              "agencyName": "X"}, ctx)


@pytest.mark.asyncio
async def test_search_enrichment_is_never_refused(stored_index):
    ctx = AgentContext(org_id=ORG)
    ctx.stored_only = True
    result = await dispatch("search_enrichment",
                            {"question": "weather in Riyadh on 4 September 2026"}, ctx)
    assert result["matches"]


@pytest.mark.asyncio
async def test_verify_fails_a_fetch_that_got_past_the_loop(stored_index):
    """Belt and braces: if a fetch is ever recorded on a restricted turn, the
    run does not pass verification."""
    from runtime import ToolCall
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    ctx.stored_only = True
    ctx.tool_calls.append(ToolCall(name="enrich_destination", args={"city": "Riyadh"},
                                   result={"city": "Riyadh", "domains": {}}))
    ctx.remember("last_answer", "Riyadh will be 28-40°C.")
    verdict = await agent.verify(ctx)

    assert verdict.passed is False
    assert any("restricted to stored enrichment" in issue for issue in verdict.issues)
