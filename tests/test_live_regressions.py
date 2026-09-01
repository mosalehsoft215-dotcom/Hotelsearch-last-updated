"""Eight proven live regressions, each pinned by the scenario that produced it.

The one thing they share: something was filled in where nothing had been
established — an age, a room count, a date, a set of cards from a previous turn,
a provider's stack trace. Each check below is about refusing to supply it.
"""
import pytest

import web_tools
from agents.hotel_search_agent import (
    DUPLICATE_HOTEL_THRESHOLD, HotelSearchAgent, _duplicated_hotel_list,
    _invented_child_ages, _uncovered_days,
)
from enrichment_index import EnrichmentIndex, SqliteVectorStore
from runtime import (
    MEM_USER_MESSAGE, AgentContext, LLMResponse, LLMToolCall, ToolCall, user_facing_error,
)
from web_enrich import Cache, Claim, Enricher, Enrichment, Source

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)

    async def complete(self, messages, tools=None):
        return self.replies.pop(0) if self.replies else LLMResponse(content="done")


def call(name, **args):
    return LLMToolCall(id=f"c-{name}", name=name, arguments=args)


def hotel(name, total, per_night=None):
    return {"hotelName": name, "hotelCode": name[:3].upper(), "categoryCode": "3",
            "price": {"totalPrice": total, "currency": "USD"},
            "location": {"city": "Riyadh", "country": "Saudi Arabia"},
            "board": "Room Only", "pricePerNight": per_night or round(total / 3, 2)}


def search_result(*rows):
    return {"uuid": "U1", "count": len(rows), "nights": 3, "hotels": list(rows)}


async def verdict(answer, *calls, asked=""):
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.extend(calls)
    ctx.turn_start = 0
    ctx.remember(MEM_USER_MESSAGE, asked)
    ctx.remember("last_answer", answer)
    return await HotelSearchAgent().verify(ctx)


# ---- 1. never invent a child's age ----

CHILD_PROMPTS = [
    "Find a hotel in Riyadh 10-13 September 2026 for 2 adults and 1 child",
    "Riyadh 10-13 Sep 2026, 1 room, 2 adults, 2 children",
    "I need a family room in Riyadh next week with our kids",
    "Hotel in Riyadh for 2 adults and an infant, 10-13 September 2026",
]


@pytest.mark.parametrize("asked", CHILD_PROMPTS)
def test_a_child_age_the_request_never_gave_is_flagged(asked):
    """The supplier prices on a child's age and a room may not be allowed to
    hold the child, so a guess produces a real price for the wrong booking."""
    calls = [ToolCall(name="search_hotel_availability",
                      args={"city": "Riyadh", "childrenAges": [8]},
                      result=search_result(hotel("Alpha", 300)))]
    assert _invented_child_ages(asked, calls) == [8]


@pytest.mark.parametrize("asked", [
    "Riyadh 10-13 Sep 2026, 2 adults and 1 child aged 8",
    "Riyadh, 2 adults + 2 children, ages 6 and 9",
    "2 adults and one child, 8 years old",
    "Riyadh for 2 adults and a child (age 8)",
])
def test_an_age_the_request_did_give_is_not_flagged(asked):
    calls = [ToolCall(name="search_hotel_availability",
                      args={"city": "Riyadh", "childrenAges": [8]},
                      result=search_result(hotel("Alpha", 300)))]
    assert _invented_child_ages(asked, calls) == []


def test_bare_numbers_in_the_request_count_as_the_ages():
    """"two children, 6 and 9" states them without the word "age"."""
    calls = [ToolCall(name="search_hotel_availability",
                      args={"city": "Riyadh", "childrenAges": [6, 9]},
                      result=search_result(hotel("Alpha", 300)))]
    assert _invented_child_ages("Riyadh, 2 adults, two children, 6 and 9", calls) == []


def test_ages_inside_occupancies_are_checked_too():
    calls = [ToolCall(name="search_hotel_availability",
                      args={"city": "Riyadh",
                            "occupancies": [{"adults": 2, "childrenAges": [4]}]},
                      result=search_result(hotel("Alpha", 300)))]
    assert _invented_child_ages("Riyadh for 2 adults and a child", calls) == [4]


def test_a_request_with_no_children_is_never_flagged():
    calls = [ToolCall(name="search_hotel_availability",
                      args={"city": "Riyadh", "childrenAges": []},
                      result=search_result(hotel("Alpha", 300)))]
    assert _invented_child_ages("Riyadh 10-13 Sep 2026, 1 room 2 adults", calls) == []


@pytest.mark.asyncio
async def test_verify_fails_a_search_that_guessed_a_child_age():
    result = await verdict(
        "Alpha is cheapest at 300 USD for 3 nights.",
        ToolCall(name="search_hotel_availability",
                 args={"city": "Riyadh", "childrenAges": [8]},
                 result=search_result(hotel("Alpha", 300))),
        asked="Find a hotel in Riyadh 10-13 September 2026 for 2 adults and 1 child")
    assert result.passed is False
    assert any("child age" in i for i in result.issues)


@pytest.mark.asyncio
async def test_asking_for_the_ages_and_stopping_passes():
    """The correct behaviour: no search tool ran at all."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(LLMResponse(content="How old is each child? I need each age "
                                       "before I can search — the price depends on it."))
    result = await agent.run(
        ctx, "Find a hotel in Riyadh 10-13 September 2026 for 2 adults and 1 child",
        llm, max_iterations=3)

    assert ctx.tool_calls == []
    assert result.blocks is None
    assert result.verification.passed, result.verification.issues


# ---- 2. never guess an ambiguous room count ----

@pytest.mark.asyncio
@pytest.mark.parametrize("asked", [
    "Riyadh 10-13 September 2026, 1 or 2 rooms, 2 adults each",
    "Book me 2-3 rooms in Riyadh for 10-13 Sep 2026",
    "Riyadh 10-13 Sep 2026: rooms 1 or 2",
    "I need 1/2 rooms in Riyadh",
])
async def test_searching_on_an_ambiguous_room_count_is_flagged(asked):
    result = await verdict(
        "Alpha is cheapest at 300 USD.",
        ToolCall(name="search_hotel_availability", args={"city": "Riyadh", "roomCount": 1},
                 result=search_result(hotel("Alpha", 300))),
        asked=asked)
    assert result.passed is False
    assert any("room count" in i for i in result.issues)


@pytest.mark.asyncio
async def test_a_date_range_is_not_read_as_an_ambiguous_room_count():
    """"10-13 September" is a stay, not a choice of rooms."""
    result = await verdict(
        "Alpha is cheapest at 300 USD.",
        ToolCall(name="search_hotel_availability", args={"city": "Riyadh", "roomCount": 1},
                 result=search_result(hotel("Alpha", 300))),
        asked="Riyadh 10-13 September 2026, 1 room, 2 adults")
    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_asking_which_room_count_and_stopping_passes():
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(LLMResponse(content="One room or two? The total depends on it, "
                                       "so tell me which and I will search."))
    result = await agent.run(ctx, "Riyadh 10-13 September 2026, 1 or 2 rooms",
                             llm, max_iterations=3)
    assert ctx.tool_calls == []
    assert result.verification.passed, result.verification.issues


# ---- 3. blocks belong to the turn that produced them ----

@pytest.fixture
def stored_weather(monkeypatch):
    index = EnrichmentIndex(SqliteVectorStore(":memory:"))
    enrichment = Enrichment(subject="Riyadh", domain="weather",
                            entity_type="city", entity_ref="Riyadh")
    source = Source(url="https://api.open-meteo.com/v1/forecast?x=1",
                    title="Open-Meteo forecast", tier="other")
    enrichment.claims = [Claim(domain="weather", field_name="forecast_2026-09-11",
                               value="28.4–40.4°C, 0 mm rain", sources=[source])]
    index.add(enrichment)
    monkeypatch.setattr(web_tools, "_index", index)
    monkeypatch.setattr(web_tools, "_enricher", Enricher([], Cache(), index=index))
    return index


@pytest.mark.asyncio
async def test_a_weather_turn_after_a_hotel_search_carries_no_hotel_cards(
        stored_weather, fake_hasura):
    """The live regression: turn 2 was a weather question and came back with all
    five hotel cards from turn 1, because blocks were built from the whole
    session's tool calls instead of this turn's."""
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "CTA", "title": "Riyadh", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "hotels": [hotel("Alpha", 300), hotel("Beta", 320)]}

    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)          # one context, as a session has

    first = await agent.run(ctx, "Find a hotel in Riyadh 10-13 September 2026", Scripted(
        LLMResponse(tool_calls=[call("search_hotel_availability", city="Riyadh",
                                     checkIn="2026-09-10", checkOut="2026-09-13")]),
        LLMResponse(content="Two options; Alpha is the cheaper.")), max_iterations=4)
    assert first.blocks is not None and len(first.blocks) == 2

    second = await agent.run(ctx, "What is the weather there on 11 September 2026?",
                             Scripted(
        LLMResponse(tool_calls=[call("search_enrichment",
                                     question="weather in Riyadh on 11 September 2026")]),
        LLMResponse(content="On 11 Sep 2026: 28.4–40.4°C, 0 mm rain.")),
                             max_iterations=4)

    assert second.blocks is None, "hotel cards leaked into a weather turn"
    assert ctx.tool_calls[ctx.turn_start:] and all(
        c.name == "search_enrichment" for c in ctx.tool_calls[ctx.turn_start:])
    assert second.verification.passed, second.verification.issues


@pytest.mark.asyncio
async def test_a_second_hotel_search_shows_only_its_own_hotels(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "CTA", "title": "Riyadh", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)

    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "hotels": [hotel("Alpha", 300)]}
    await agent.run(ctx, "Find a hotel in Riyadh 10-13 September 2026", Scripted(
        LLMResponse(tool_calls=[call("search_hotel_availability", city="Riyadh",
                                     checkIn="2026-09-10", checkOut="2026-09-13")]),
        LLMResponse(content="Alpha at 300 USD.")), max_iterations=4)

    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "hotels": [hotel("Gamma", 400)]}
    second = await agent.run(ctx, "Now try 20-23 September 2026", Scripted(
        LLMResponse(tool_calls=[call("search_hotel_availability", city="Riyadh",
                                     checkIn="2026-09-20", checkOut="2026-09-23")]),
        LLMResponse(content="Gamma at 400 USD.")), max_iterations=4)

    assert [b.hotel_name for b in second.blocks] == ["Gamma"]


# ---- 4. the prose and the cards divide the work ----

def test_a_full_hotel_list_repeated_in_the_prose_is_flagged():
    calls = [ToolCall(name="search_hotel_availability", args={},
                      result=search_result(hotel("Alpha", 300), hotel("Beta", 320),
                                           hotel("Gamma", 340)))]
    answer = ("- **Alpha** — 300 USD total\n"
              "- **Beta** — 320 USD total\n"
              "- **Gamma** — 340 USD total")
    assert _duplicated_hotel_list(answer, calls) == ["Alpha", "Beta", "Gamma"]


def test_a_summary_that_names_one_hotel_and_its_price_is_fine():
    calls = [ToolCall(name="search_hotel_availability", args={},
                      result=search_result(hotel("Alpha", 300), hotel("Beta", 320),
                                           hotel("Gamma", 340)))]
    answer = ("Three options, all 3-star and room-only. Alpha is the cheapest at "
              "300 USD total; the cards below carry the rest.")
    assert _duplicated_hotel_list(answer, calls) == []


def test_fewer_than_the_threshold_is_never_duplication():
    calls = [ToolCall(name="search_hotel_availability", args={},
                      result=search_result(hotel("Alpha", 300), hotel("Beta", 320)))]
    answer = "- Alpha — 300 USD\n- Beta — 320 USD"
    assert _duplicated_hotel_list(answer, calls) == []
    assert DUPLICATE_HOTEL_THRESHOLD == 3


@pytest.mark.asyncio
async def test_a_duplicated_hotel_list_is_noted_without_failing_the_answer():
    """A note, not an issue. The prose is accurate — it is the card list written
    out twice — and `passed` has always meant "this may be false": an invented
    price, a swapped date, a rate called confirmed that never was. Filing
    verbosity under the same badge would teach the reader to ignore it."""
    result = await verdict(
        "- **Alpha** — 300 USD\n- **Beta** — 320 USD\n- **Gamma** — 340 USD",
        ToolCall(name="search_hotel_availability", args={},
                 result=search_result(hotel("Alpha", 300), hotel("Beta", 320),
                                      hotel("Gamma", 340))),
        asked="Find hotels in Riyadh 10-13 September 2026")
    assert result.passed is True, result.issues
    assert result.issues == []
    assert any("already have cards" in n for n in result.notes)


@pytest.mark.asyncio
async def test_a_summarising_answer_is_not_even_noted():
    result = await verdict(
        "Three options, all 3-star and room-only, from 300 to 340 USD for the "
        "three nights. Alpha is the cheapest; none is refundable.",
        ToolCall(name="search_hotel_availability", args={},
                 result=search_result(hotel("Alpha", 300), hotel("Beta", 320),
                                      hotel("Gamma", 340))),
        asked="Find hotels in Riyadh 10-13 September 2026")
    assert result.passed is True and result.notes == []


@pytest.mark.asyncio
async def test_a_real_fault_still_fails_rather_than_being_noted():
    """The badge keeps its meaning. An invented price is an issue, not a note —
    and this is the case that repaired money regex now actually catches."""
    result = await verdict(
        "Fiction Hotel is available at 999 USD total.",
        ToolCall(name="search_hotel_availability", args={},
                 result=search_result(hotel("Alpha", 300))),
        asked="Find hotels in Riyadh 10-13 September 2026")
    assert result.passed is False
    assert any("999" in i for i in result.issues)


# ---- 5. every weather date shown must be one the source returned ----

def weather_call(days):
    return ToolCall(
        name="search_enrichment", args={"question": "weather in Riyadh"},
        result={"matches": [
            {"domain": "weather", "field": f"forecast_{day}", "value": value,
             "entity_ref": "Riyadh", "subject": "Riyadh",
             "sources": [{"url": "https://api.open-meteo.com/v1/forecast",
                          "title": "Open-Meteo forecast", "tier": "other"}],
             "observed_at": "2026-09-01T00:00:00+00:00", "is_stale": False,
             "match": 0.9}
            for day, value in days.items()]})


WINDOW = {"2026-09-10": "27.2–39.8°C, 0 mm rain",
          "2026-09-11": "28.4–40.4°C, 0 mm rain"}


@pytest.mark.asyncio
async def test_a_date_outside_the_source_window_is_flagged():
    """The gap the alignment check could not see: a real figure reused under a
    date nothing covered. Both the membership test and the per-day test passed,
    because the day was not in the source at all and was therefore skipped."""
    result = await verdict(
        "10 Sep: 27.2–39.8°C, 0 mm rain\n"
        "11 Sep: 28.4–40.4°C, 0 mm rain\n"
        "12 Sep: 28.4–40.4°C, 0 mm rain",
        weather_call(WINDOW), asked="Weather in Riyadh 10-12 September 2026")
    assert result.passed is False
    assert any("no forecast covered" in i for i in result.issues)
    assert any("09-12" in i for i in result.issues)


@pytest.mark.asyncio
async def test_reporting_the_uncovered_date_as_missing_passes():
    result = await verdict(
        "10 Sep: 27.2–39.8°C, 0 mm rain\n"
        "11 Sep: 28.4–40.4°C, 0 mm rain\n"
        "12 Sep is outside the window the forecast returned, so I have no figures "
        "for it.",
        weather_call(WINDOW), asked="Weather in Riyadh 10-12 September 2026")
    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_each_covered_date_keeps_its_own_figures():
    swapped = await verdict(
        "10 Sep: 28.4–40.4°C, 0 mm rain\n11 Sep: 27.2–39.8°C, 0 mm rain",
        weather_call(WINDOW), asked="Weather in Riyadh 10-11 September 2026")
    assert swapped.passed is False
    assert any("another day's figures" in i for i in swapped.issues)

    aligned = await verdict(
        "10 Sep: 27.2–39.8°C, 0 mm rain\n11 Sep: 28.4–40.4°C, 0 mm rain",
        weather_call(WINDOW), asked="Weather in Riyadh 10-11 September 2026")
    assert aligned.passed is True, aligned.issues


def test_the_uncovered_check_ignores_a_line_with_no_figures():
    calls = [weather_call(WINDOW)]
    assert _uncovered_days("No data for 19 December 2026.", calls) == []
    assert _uncovered_days("19 December 2026: 30°C", calls) == ["12-19"]


# ---- 6. Open-Meteo is a forecast vendor, not an authority ----

@pytest.mark.asyncio
async def test_open_meteo_is_not_labelled_official():
    import httpx
    from web_enrich import OpenMeteo

    def handler(request):
        if "geocoding" in str(request.url):
            return httpx.Response(200, json={"results": [
                {"name": "Riyadh", "country": "Saudi Arabia",
                 "latitude": 24.7, "longitude": 46.7}]})
        return httpx.Response(200, json={"daily": {
            "time": ["2026-09-10"], "temperature_2m_max": [39.8],
            "temperature_2m_min": [27.2], "precipitation_sum": [0.0]}})

    claims = await OpenMeteo(transport=httpx.MockTransport(handler)).fetch(
        "Riyadh", "weather", {"check_in": "2026-09-10", "check_out": "2026-09-10"})

    for claim in claims:
        assert claim.sources[0].tier == "other", "a forecast vendor is not official"
        assert claim.sources[0].tier != "gov"
        assert claim.authority == "third_party"
        assert claim.sources[0].title == "Open-Meteo forecast"


@pytest.mark.asyncio
async def test_a_weather_answer_cannot_claim_an_official_advisory():
    """The word stays reserved for a government, and weather never earns it."""
    result = await verdict(
        "The official government travel advisory says it will be 39.8°C.",
        weather_call(WINDOW), asked="Weather in Riyadh")
    assert result.passed is False
    assert any("official government travel advisory" in i for i in result.issues)


# ---- 8. a provider's own error text never reaches the reader ----

@pytest.mark.parametrize("raw,expect", [
    ('LLMError: api.mistral.ai HTTP 503: {"object":"error","message":"Service '
     'unavailable","type":"server_error"}', "busy or briefly unavailable"),
    ('LLMError: api.mistral.ai HTTP 429: {"message":"Rate limit exceeded",'
     '"code":"1300"}', "busy or briefly unavailable"),
    ("LLMError: openrouter.ai HTTP 402 Insufficient credits", "no credit left"),
    ("LLMError: in-flight budget exhausted for this key", "in flight"),
    ("LLMError: api.groq.com HTTP 401 Invalid API Key", "key was rejected"),
    ("LLMError: openrouter.ai HTTP 404 no such model: bogus/model", "cannot reach"),
    ("KeyError: 'some_internal_field'", "Something went wrong"),
    ("ValueError: unexpected None in _build_selection", "Something went wrong"),
])
def test_a_provider_failure_is_reported_without_its_internals(raw, expect):
    clean = user_facing_error(raw)
    assert expect in clean
    for leak in ("mistral", "groq", "openrouter", "HTTP", "503", "429", "401", "404",
                 "{", "}", "LLMError", "KeyError", "ValueError", "_build_selection"):
        assert leak not in clean, f"{leak!r} leaked into {clean!r}"


def test_every_clean_message_says_what_to_do_next():
    from runtime import _CLEAN_ERRORS, _GENERIC_ERROR
    for _, message in _CLEAN_ERRORS:
        assert message.endswith(".")
        assert any(hint in message.lower()
                   for hint in ("send", "pick another", "wait")), message
    assert "Try again" in _GENERIC_ERROR


@pytest.mark.asyncio
async def test_the_chat_endpoint_returns_the_clean_message(monkeypatch, fake_hasura):
    from fastapi.testclient import TestClient
    import api
    from runtime import LLMError

    class Failing:
        async def complete(self, messages, tools=None):
            raise LLMError('api.mistral.ai HTTP 503: {"object":"error",'
                           '"message":"Service unavailable"}')

        async def aclose(self):
            return None

    monkeypatch.setattr(api, "build_llm", lambda *_a, **_k: Failing())
    body = TestClient(api.app).post("/chat", json={"message": "hi", "org_id": ORG}).json()

    assert "busy or briefly unavailable" in body["error"]
    for leak in ("mistral", "HTTP", "503", "LLMError", "{"):
        assert leak not in body["error"], leak
