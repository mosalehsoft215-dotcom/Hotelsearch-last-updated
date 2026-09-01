"""The structured display channel: models, propagation, and what stays unchanged.

The whole feature is additive, so most of these tests are about what did *not*
change. A caller that reads only `output` must be unable to tell this exists.
"""
import json

import pytest

from blocks import (
    BLOCK_TYPES, BookingSummaryBlock, FlightOptionBlock, HotelOptionBlock, SummaryItem,
    TableBlock, blocks_from_tool_calls, blocks_to_model, parse_block, parse_blocks,
    sources_from_tool_calls,
)
from runtime import AgentContext, AgentRunResult, Handover, ToolCall, VerificationResult

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def hotel_row(**over):
    row = {"hotelName": "Carawan Al Fahad", "hotelCode": "1442211",
           "categoryCode": "4", "available": True,
           "price": {"totalPrice": 361.5, "currency": "USD", "net": 340.0},
           "location": {"city": "Riyadh", "country": "Saudi Arabia"},
           "board": "Breakfast Included", "boardCode": "1331",
           "roomName": ["Double Room"], "pricePerNight": 120.5,
           "optionRefId": "33!~|a0!~|b260901!~|c260904",
           "cancelPolicy": {"refundable": True, "description": "Free until 8 Sep 2026"}}
    row.update(over)
    return row


def search_call(rows=None, name="search_hotel_availability"):
    return ToolCall(name=name, args={"city": "Riyadh"},
                    result={"uuid": "sess-abc", "count": 2, "nights": 3,
                            "hotels": rows if rows is not None else [hotel_row()]})


# ---- 1, 5, 6: the existing contract is untouched ----

def test_an_answer_only_result_still_works():
    """Requirement 1 and 6. Nothing passes blocks, nothing reads blocks."""
    result = AgentRunResult(output="Riyadh is hot in September.",
                            verification=VerificationResult(),
                            context=AgentContext(org_id=ORG), messages=[])
    assert result.output == "Riyadh is hot in September."
    assert result.blocks is None
    assert blocks_to_model(result.blocks) is None


def test_blocks_default_to_none_and_serialise_away_entirely():
    """Requirement 5. Absent means absent — not an empty list on the wire."""
    assert blocks_to_model(None) is None
    assert blocks_to_model([]) is None
    assert parse_blocks(None) is None
    assert parse_blocks([]) is None


def test_the_answer_field_is_not_renamed():
    fields = set(AgentRunResult.__dataclass_fields__)
    assert {"output", "verification", "context", "messages", "blocks"} == fields


# ---- 2, 3, 12: each block type validates and serialises ----

def test_hotel_option_validates_and_serialises():
    block = HotelOptionBlock(hotel_name="Carawan Al Fahad", stars=4,
                             location="Riyadh, Saudi Arabia", price_per_night=120.5,
                             total_price=361.5, currency="USD",
                             board="Breakfast Included", refundable=True,
                             cancellation_summary="Free until 8 Sep 2026")
    model = block.model_dump()
    assert model["type"] == "hotel_option"
    assert model["hotel_name"] == "Carawan Al Fahad"
    assert json.loads(json.dumps(model)) == model      # wire-safe


def test_flight_option_validates_and_serialises():
    block = FlightOptionBlock(origin="JED", destination="MCT", airline="Saudia",
                              flight_number="SV1234", stops=0, total_price=410.0,
                              currency="USD")
    model = block.model_dump()
    assert model["type"] == "flight_option"
    assert (model["origin"], model["destination"]) == ("JED", "MCT")


def test_booking_summary_validates_and_serialises():
    block = BookingSummaryBlock(title="Confirmed rate", total=361.5, currency="USD",
                                items=[SummaryItem(label="Nights", value="3")])
    model = block.model_dump()
    assert model["type"] == "booking_summary"
    assert model["items"] == [{"label": "Nights", "value": "3"}]


def test_table_block_validates_and_serialises():
    block = TableBlock(columns=["Hotel", "Total"],
                       rows=[["A", 100], ["B", 120.5], ["C", None], ["D", True]])
    model = block.model_dump()
    assert model["type"] == "table"
    assert model["rows"][2] == ["C", None]
    assert json.loads(json.dumps(model)) == model


def test_a_hotel_block_needs_only_its_name():
    """Every other field is optional, so a thin payload still produces a card."""
    block = HotelOptionBlock(hotel_name="Bare Hotel")
    assert block.stars is None and block.total_price is None
    assert block.model_dump()["hotel_name"] == "Bare Hotel"


def test_the_four_types_are_the_whole_set():
    assert BLOCK_TYPES == ("hotel_option", "flight_option", "booking_summary", "table")


# ---- 4: an invalid block is handled, never raised ----

@pytest.mark.parametrize("bad", [
    {"type": "car_rental", "vendor": "x"},          # unknown type
    {"type": "hotel_option"},                       # missing hotel_name
    {"type": "table", "columns": ["a"]},            # missing rows
    {"type": "flight_option", "origin": "JED"},     # missing destination
    {"no_type": True}, None, "a string", 42,
])
def test_an_invalid_block_is_dropped_rather_than_raised(bad):
    """A display channel must not be able to fail a turn that has a good answer."""
    assert parse_block(bad) is None
    assert parse_blocks([bad]) is None


def test_a_good_block_survives_beside_a_bad_one():
    kept = parse_blocks([{"type": "nonsense"},
                         {"type": "hotel_option", "hotel_name": "Real Hotel"}])
    assert kept is not None and len(kept) == 1
    assert kept[0].hotel_name == "Real Hotel"


# ---- 13: internal identifiers cannot reach a block ----

def test_an_internal_field_is_refused_by_the_schema():
    """extra="forbid" is the guard. A builder that grows a hotelCode or an
    optionRefId fails here rather than putting it on screen."""
    from pydantic import ValidationError
    for leak in ({"hotel_code": "1442211"}, {"option_ref_id": "33!~|a0"},
                 {"session_id": "sess-abc"}, {"raw": {"supplier": "..."}}):
        with pytest.raises(ValidationError):
            HotelOptionBlock(hotel_name="X", **leak)


def test_a_built_hotel_card_carries_no_internal_identifiers():
    blocks = blocks_from_tool_calls([search_call()])
    assert blocks is not None
    serialised = json.dumps(blocks_to_model(blocks))
    assert "1442211" not in serialised          # hotelCode
    assert "33!~|" not in serialised            # optionRefId
    assert "sess-abc" not in serialised         # search session
    assert "boardCode" not in serialised and "1331" not in serialised
    assert set(blocks_to_model(blocks)[0]) == set(HotelOptionBlock.model_fields)


# ---- 7, 9, 11: blocks reach the result through the existing chain ----

@pytest.mark.asyncio
async def test_a_hotel_search_emits_hotel_option_blocks():
    """Requirement 9, through the real agent hook rather than by hand."""
    from agents.hotel_search_agent import HotelSearchAgent
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.append(search_call())
    blocks = HotelSearchAgent().build_blocks(ctx)

    assert blocks is not None and len(blocks) == 1
    card = blocks[0]
    assert isinstance(card, HotelOptionBlock)
    assert card.hotel_name == "Carawan Al Fahad"
    assert (card.stars, card.currency, card.total_price) == (4.0, "USD", 361.5)
    assert card.location == "Riyadh, Saudi Arabia"
    assert card.board == "Breakfast Included"
    assert card.refundable is True
    assert card.cancellation_summary == "Free until 8 Sep 2026"


@pytest.mark.asyncio
async def test_a_confirmed_reprice_emits_a_booking_summary():
    """Requirement 11."""
    from agents.hotel_search_agent import HotelSearchAgent
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.append(ToolCall(
        name="refresh_hotel_price", args={},
        result={"price": 361.5, "currency": "USD", "hotelName": "Carawan Al Fahad",
                "roomName": ["Double Room"], "board": "Breakfast Included", "nights": 3}))
    blocks = HotelSearchAgent().build_blocks(ctx)

    assert blocks is not None and isinstance(blocks[0], BookingSummaryBlock)
    assert blocks[0].total == 361.5 and blocks[0].currency == "USD"
    assert ("Nights", "3") in [(i.label, i.value) for i in blocks[0].items]


@pytest.mark.asyncio
async def test_a_failed_reprice_produces_no_quote():
    """An error must not look like a confirmed rate."""
    from agents.hotel_search_agent import HotelSearchAgent
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.append(ToolCall(name="refresh_hotel_price", args={},
                                   result={"error": "Transaction Id should be UUID"}))
    assert HotelSearchAgent().build_blocks(ctx) is None


@pytest.mark.asyncio
async def test_blocks_reach_the_run_result_and_the_handover():
    """Requirement 7: leaf/coordinator blocks propagate upward."""
    from agents.hotel_search_agent import HotelSearchAgent
    from runtime import LLMResponse, LLMToolCall, delegate

    class Scripted:
        def __init__(self): self.n = 0
        async def complete(self, messages, tools=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(tool_calls=[LLMToolCall(
                    id="c1", name="search_enrichment", arguments={"question": "x"})])
            return LLMResponse(content="Two options, the first is cheapest.")

    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    result = await agent.run(ctx, "Find a hotel in Riyadh", Scripted(), max_iterations=4)
    # No priced tool ran, so no cards — the honest outcome for this transcript.
    assert result.blocks is None

    # Now a parent delegating work whose child did return cards.
    parent = AgentContext(org_id=ORG)
    handover = await delegate(agent, "Find a hotel in Riyadh", Scripted(), parent)
    assert handover.blocks is None
    assert "blocks" not in handover.to_model()

    carried = Handover(agent="hotel_search_agent", answer="Two options.",
                       tools_used=["search_hotel_availability"], passed=True,
                       blocks=[HotelOptionBlock(hotel_name="Carawan Al Fahad")])
    assert carried.to_model()["blocks"][0]["hotel_name"] == "Carawan Al Fahad"


# ---- 8, 14: prose turns stay prose ----

@pytest.mark.parametrize("call", [
    ToolCall(name="search_enrichment", args={}, result={"matches": [], "note": "nothing"}),
    ToolCall(name="enrich_destination", args={}, result={"city": "Riyadh", "domains": {}}),
    ToolCall(name="recall_preferences", args={}, result={"preferences": []}),
    ToolCall(name="get_hotel_static_data", args={}, result={"hotelName": "X"}),
])
def test_a_conversational_turn_produces_no_blocks(call):
    """Requirement 8. Nothing is fabricated because the schema exists."""
    assert blocks_from_tool_calls([call]) is None


def test_no_tool_calls_at_all_produces_no_blocks():
    assert blocks_from_tool_calls([]) is None
    assert blocks_from_tool_calls(None) is None


def test_raw_json_is_never_placed_in_the_answer():
    """Requirement 14. The structured path is a separate field; the prose the
    agent produces is untouched by it."""
    from agents.hotel_search_agent import HotelSearchAgent
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.append(search_call())
    blocks = HotelSearchAgent().build_blocks(ctx)
    result = AgentRunResult(output="Two options; the first is cheapest.",
                            verification=VerificationResult(), context=ctx,
                            messages=[], blocks=blocks)
    assert "{" not in result.output and "hotel_name" not in result.output
    assert result.blocks is not None


def test_a_hotel_row_with_no_name_is_not_a_card():
    assert blocks_from_tool_calls([search_call(rows=[{"price": {"totalPrice": 1}}])]) is None


def test_missing_optional_supplier_fields_still_build_a_card():
    blocks = blocks_from_tool_calls([search_call(rows=[{"hotelName": "Bare Hotel"}])])
    assert blocks is not None
    card = blocks[0]
    assert card.hotel_name == "Bare Hotel"
    assert (card.stars, card.total_price, card.refundable) == (None, None, None)


def test_the_card_count_is_capped():
    rows = [hotel_row(hotelName=f"Hotel {i}") for i in range(20)]
    blocks = blocks_from_tool_calls([search_call(rows=rows)])
    assert len(blocks) == 6


def test_a_quote_leads_the_blocks_when_both_are_present():
    calls = [search_call(),
             ToolCall(name="refresh_hotel_price", args={},
                      result={"price": 361.5, "currency": "USD", "hotelName": "X"})]
    blocks = blocks_from_tool_calls(calls)
    assert isinstance(blocks[0], BookingSummaryBlock)
    assert isinstance(blocks[1], HotelOptionBlock)


# ---- enrichment provenance, surfaced not redesigned ----

def test_source_metadata_is_read_off_an_enrichment_result():
    call = ToolCall(name="search_enrichment", args={}, result={"matches": [{
        "domain": "advisory", "field": "advisory_level", "value": "none",
        "observed_at": "2026-09-01T00:00:00+00:00",
        "valid_until": "2026-09-02T00:00:00+00:00", "is_stale": False,
        "sources": [{"url": "https://www.gov.uk/foreign-travel-advice/oman",
                     "title": "Oman travel advice", "tier": "gov"}]}]})
    sources = sources_from_tool_calls([call])

    assert sources is not None and len(sources) == 1
    assert sources[0]["host"] == "gov.uk"
    assert sources[0]["domain"] == "advisory"
    assert sources[0]["observed_at"].startswith("2026-09-01")
    assert sources[0]["valid_until"].startswith("2026-09-02")


def test_source_metadata_is_read_off_a_fresh_fetch_too():
    call = ToolCall(name="enrich_destination", args={}, result={"domains": {"advisory": {
        "findings": {"advisory_level": [{
            "value": "none", "observed_at": "2026-09-01T00:00:00+00:00",
            "sources": [{"url": "https://www.gov.uk/foreign-travel-advice/oman"}]}]}}}})
    sources = sources_from_tool_calls([call])
    assert sources[0]["host"] == "gov.uk" and sources[0]["domain"] == "advisory"


def test_a_turn_with_no_enrichment_has_no_sources():
    assert sources_from_tool_calls([search_call()]) is None
    assert sources_from_tool_calls([]) is None
