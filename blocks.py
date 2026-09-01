"""The structured display channel that runs alongside the prose answer.

An answer is markdown for a person to read. A block is the same information in a
shape a page can lay out — a hotel card, a summary, a table. The two are separate
fields on purpose:

* nothing parses JSON back out of the prose, and no page regex-matches a table
  out of a sentence to rebuild a card;
* the model never has to emit JSON into text a person is going to read;
* a caller that only wants `output` is unaffected, because `blocks` is optional
  and absent by default.

Blocks are built from what the tools actually returned, in `blocks_from_tool_calls`
— not from the model's summary of it. A card is a display of supplier data, so it
has to come from the supplier payload rather than from prose about it.

`extra="forbid"` is the point of the schema, not a detail. A display block is a
contract with the page, so a builder that grows an extra field — a hotel code, an
option reference, a session id — fails validation here instead of putting an
internal identifier on screen.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Iterable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

logger = logging.getLogger("tripon.blocks")


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HotelOptionBlock(_Block):
    """One bookable choice at one hotel, as a card."""
    type: Literal["hotel_option"] = "hotel_option"
    hotel_name: str
    stars: float | None = None
    location: str | None = None
    price_per_night: float | None = None
    total_price: float | None = None
    currency: str | None = None
    board: str | None = None
    refundable: bool | None = None
    cancellation_summary: str | None = None


class FlightOptionBlock(_Block):
    """One flight, as a card. Defined and rendered; nothing emits it yet, because
    this repo has no flight tools — `flights` appears in GRANTED_MODULES as a
    permission name only. The shape is here so the path exists the day one is
    added, rather than being invented then."""
    type: Literal["flight_option"] = "flight_option"
    airline: str | None = None
    flight_number: str | None = None
    origin: str
    destination: str
    departure: str | None = None
    arrival: str | None = None
    duration: str | None = None
    stops: int | None = None
    total_price: float | None = None
    currency: str | None = None


class SummaryItem(_Block):
    label: str
    value: str


class BookingSummaryBlock(_Block):
    """A confirmed or quoted total, broken into labelled lines."""
    type: Literal["booking_summary"] = "booking_summary"
    title: str
    items: list[SummaryItem] = Field(default_factory=list)
    total: float | None = None
    currency: str | None = None


# What a table cell may hold. Scalars only: a nested object in a cell is a sign
# something is being passed through rather than displayed.
Cell = Union[str, int, float, bool, None]


class TableBlock(_Block):
    """A genuine side-by-side comparison. Not a dumping ground for a payload."""
    type: Literal["table"] = "table"
    columns: list[str]
    rows: list[list[Cell]]


Block = Annotated[
    Union[HotelOptionBlock, FlightOptionBlock, BookingSummaryBlock, TableBlock],
    Field(discriminator="type"),
]

BLOCK_TYPES = ("hotel_option", "flight_option", "booking_summary", "table")

_ONE = TypeAdapter(Block)


def parse_block(raw: Any) -> Block | None:
    """One block, or None if it is not one. Unknown `type` and malformed fields
    both land here, and both are dropped rather than raised: a display channel
    must not be able to fail a turn that already has a good answer."""
    try:
        return _ONE.validate_python(raw)
    except ValidationError as exc:
        logger.warning("dropped an invalid display block: %s",
                       exc.errors()[0].get("msg") if exc.errors() else exc)
        return None


def parse_blocks(raw: Any) -> list[Block] | None:
    """A list of blocks from loose input, or None when there is nothing usable.
    None rather than [] so the field stays absent end to end — an empty list on
    the wire reads as "there were blocks and they are all gone"."""
    if not raw:
        return None
    if isinstance(raw, dict):
        raw = [raw]
    kept = [block for block in (parse_block(item) for item in raw) if block is not None]
    return kept or None


def blocks_to_model(blocks: Iterable[Block] | None) -> list[dict[str, Any]] | None:
    """What goes on the wire. None stays None so the API shape is unchanged for
    every answer that has no structured data."""
    if not blocks:
        return None
    return [block.model_dump() for block in blocks]


# --- building blocks from what the tools returned -----------------------------

# Tools whose payload carries bookable choices. Read for display only; the
# supplier still owns the numbers and nothing here is written to the enrichment
# index — live price and availability stay on the supplier path.
_HOTEL_LIST_TOOLS = ("search_hotel_availability", "get_hotel_search_results")
_HOTEL_OPTION_TOOLS = ("get_hotel_availability_options", "get_hotel_options")
_QUOTE_TOOLS = ("refresh_hotel_price",)

MAX_HOTEL_CARDS = 6


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stars_of(hotel: dict[str, Any]) -> float | None:
    """The star rating, from the same field the star filters read."""
    return _number(hotel.get("categoryCode"))


def _place(hotel: dict[str, Any]) -> str | None:
    spot = hotel.get("location") or {}
    if not isinstance(spot, dict):
        return None
    parts = [str(spot.get(key)) for key in ("city", "country") if spot.get(key)]
    return ", ".join(parts) or None


def _hotel_card(hotel: dict[str, Any]) -> HotelOptionBlock | None:
    name = hotel.get("hotelName") or hotel.get("name")
    if not name:
        return None                      # a card with no subject is not a card
    price = hotel.get("price") if isinstance(hotel.get("price"), dict) else {}
    policy = hotel.get("cancelPolicy") if isinstance(hotel.get("cancelPolicy"), dict) else {}
    rooms = hotel.get("roomName")
    board = hotel.get("board")
    if not board and isinstance(rooms, list) and rooms:
        board = str(rooms[0])
    return HotelOptionBlock(
        hotel_name=str(name),
        stars=_stars_of(hotel),
        location=_place(hotel),
        price_per_night=_number(hotel.get("pricePerNight")),
        total_price=_number(price.get("totalPrice") or hotel.get("totalPrice")),
        currency=(price.get("currency") or hotel.get("currency") or None),
        board=str(board) if board else None,
        refundable=policy.get("refundable") if isinstance(policy.get("refundable"), bool) else None,
        cancellation_summary=(str(policy["description"])[:200]
                              if policy.get("description") else None),
    )


def _quote_summary(result: dict[str, Any]) -> BookingSummaryBlock | None:
    """A confirmed rate, as a summary. Only when a price actually came back —
    a failed reprice has nothing to summarise and must not look like a quote."""
    total = _number(result.get("price") or result.get("totalPrice"))
    if total is None or result.get("error"):
        return None
    items: list[SummaryItem] = []
    for label, key in (("Hotel", "hotelName"), ("Room", "roomName"),
                       ("Board", "board"), ("Nights", "nights")):
        value = result.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value not in (None, ""):
            items.append(SummaryItem(label=label, value=str(value)))
    return BookingSummaryBlock(
        title="Confirmed rate",
        items=items,
        total=total,
        currency=result.get("currency") or None,
    )


def blocks_from_tool_calls(tool_calls: Iterable[Any]) -> list[Block] | None:
    """Structured cards for the run, derived from the tool payloads themselves.

    Conversational turns produce nothing here, and that is the intended outcome:
    a greeting, a weather explanation, an advisory summary or a clarifying
    question has no structured display data, so `blocks` stays None rather than
    being filled because the schema exists.
    """
    hotels: list[Block] = []
    summaries: list[Block] = []
    for call in tool_calls or []:
        name = getattr(call, "name", None)
        result = getattr(call, "result", None)
        if not isinstance(result, dict) or result.get("error"):
            continue
        if name in _HOTEL_LIST_TOOLS or name in _HOTEL_OPTION_TOOLS:
            rows = result.get("hotels") or result.get("options") or []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    card = _hotel_card(row)
                    if card is not None:
                        hotels.append(card)
        elif name in _QUOTE_TOOLS:
            summary = _quote_summary(result)
            if summary is not None:
                summaries.append(summary)
    # The quote is the answer to "lock this rate", so it leads when present.
    ordered = [*summaries, *hotels[:MAX_HOTEL_CARDS]]
    return ordered or None


# --- enrichment provenance, for display only ---------------------------------

# Read from the enrichment payloads the turn already produced. Nothing is
# fetched, nothing is indexed and retrieval is untouched — this only carries
# metadata that was already on the tool result out to the page, so an answer can
# show where a claim came from and how fresh it is.
_ENRICHMENT_TOOLS = ("enrich_hotel_info", "enrich_destination", "search_enrichment",
                     "enrich_company_facts", "enrich_agency_facts")

MAX_SOURCES = 6


def _host_of(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url or "").netloc or "").lower().removeprefix("www.")


def _add_source(seen: dict[str, dict[str, Any]], url: Any, domain: Any,
                observed_at: Any, valid_until: Any, is_stale: Any) -> None:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return
    entry = seen.setdefault(url, {"url": url, "host": _host_of(url) or url})
    if domain and not entry.get("domain"):
        entry["domain"] = str(domain)
    if observed_at and not entry.get("observed_at"):
        entry["observed_at"] = str(observed_at)
    if valid_until and not entry.get("valid_until"):
        entry["valid_until"] = str(valid_until)
    if is_stale:
        entry["is_stale"] = True


def sources_from_tool_calls(tool_calls: Iterable[Any]) -> list[dict[str, Any]] | None:
    """Where this turn's enrichment claims came from, deduplicated by url."""
    seen: dict[str, dict[str, Any]] = {}
    for call in tool_calls or []:
        if getattr(call, "name", None) not in _ENRICHMENT_TOOLS:
            continue
        result = getattr(call, "result", None)
        if not isinstance(result, dict):
            continue
        # A fresh fetch: domains -> findings -> entries, each with its sources.
        for domain, payload in (result.get("domains") or {}).items():
            if not isinstance(payload, dict):
                continue
            for entries in (payload.get("findings") or {}).values():
                for entry in entries if isinstance(entries, list) else []:
                    if not isinstance(entry, dict):
                        continue
                    for source in entry.get("sources") or []:
                        if isinstance(source, dict):
                            _add_source(seen, source.get("url"), domain,
                                        entry.get("observed_at"),
                                        entry.get("valid_until"), False)
        # A stored claim, straight off the index.
        for match in result.get("matches") or []:
            if not isinstance(match, dict):
                continue
            for source in match.get("sources") or []:
                if isinstance(source, dict):
                    _add_source(seen, source.get("url"), match.get("domain"),
                                match.get("observed_at"), match.get("valid_until"),
                                match.get("is_stale"))
    return list(seen.values())[:MAX_SOURCES] or None
