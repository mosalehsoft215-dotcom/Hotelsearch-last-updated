"""The enrichment tools an agent may call."""
from __future__ import annotations

from typing import Any

from config import get_settings
from enrichment_index import MIN_SCORE, EnrichmentIndex, mentioned_entities
from web_enrich import (
    DESTINATION_DOMAINS, DOMAINS, HOTEL_DOMAINS, Cache, Enricher, build_providers,
)

_settings = get_settings()
_index = EnrichmentIndex(path=_settings.enrichment_index_path)
_enricher = Enricher(build_providers(_settings), Cache(), index=_index)


def index_stats() -> int:
    """How many claims the index holds. A read-only window for callers outside this
    module, so nothing has to reach into the private `_index`."""
    return _index.size()


async def enrich_hotel_info(hotelName: str, city: str | None = None,
                            domains: list[str] | None = None,
                            officialSite: str | None = None,
                            pageUrl: str | None = None) -> dict[str, Any]:
    """What the outside world says about one hotel: reputation, location,
    facilities, and anything that would spoil the stay. Cited, and never a source
    of price or policy."""
    wanted = [d for d in (domains or list(HOTEL_DOMAINS)) if d in HOTEL_DOMAINS]
    subject = f"{hotelName}, {city}" if city else hotelName
    context = {"official_domains": [officialSite] if officialSite else [],
               "page_url": pageUrl}
    return {
        "hotel": hotelName,
        "city": city,
        "domains": {d: (await _enricher.enrich(subject, d, context, entity_type="hotel",
                                               entity_ref=hotelName)).to_model()
                    for d in wanted},
    }


async def enrich_destination(city: str, checkIn: str | None = None,
                             checkOut: str | None = None,
                             domains: list[str] | None = None) -> dict[str, Any]:
    """Conditions at the destination for the stay: weather for the dates, the
    current travel advice, and recent news worth knowing."""
    wanted = [d for d in (domains or list(DESTINATION_DOMAINS)) if d in DESTINATION_DOMAINS]
    context = {"check_in": checkIn, "check_out": checkOut}
    return {
        "city": city,
        "checkIn": checkIn,
        "checkOut": checkOut,
        "domains": {d: (await _enricher.enrich(city, d, context, entity_type="city",
                                               entity_ref=city)).to_model()
                    for d in wanted},
    }


def _note(question: str, matches: list[dict[str, Any]],
          stale: list[dict[str, Any]] | None = None) -> str | None:
    """Say which subject came up empty, not just that something did. Asking
    about a city the index has never fetched used to return another city's
    weather at 0.757, because the domain vocabulary matches whatever the
    question is about."""
    if matches:
        return None
    if stale:
        subject = stale[0].get("entity_ref") or stale[0].get("subject") or "this subject"
        return (f"everything held for {subject} in this domain has passed its "
                "freshness window. Fetch it again with enrich_destination or "
                "enrich_hotel_info rather than answering from it.")
    named = mentioned_entities(question)
    if named:
        return (f"nothing enriched so far covers {named[0]}. "
                "Fetch it first with enrich_destination or enrich_hotel_info.")
    return "nothing enriched so far answers this"


async def search_enrichment(question: str, limit: int = 5, subject: str | None = None,
                            domain: str | None = None, entityType: str | None = None,
                            entityRef: str | None = None,
                            minScore: float | None = None,
                            includeStale: bool = False) -> dict[str, Any]:
    """Ask in plain words across everything enrichment has already fetched, with no
    need to know the entity or the domain first. Narrow with entityType ("hotel"
    or "city"), entityRef, or domain when you do know them."""
    if domain and domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {', '.join(DOMAINS)}")
    if entityType and entityType not in ("hotel", "city"):
        raise ValueError(f"unknown entityType {entityType!r}; expected 'hotel' or 'city'")
    matches = _index.search(question, limit=limit, subject=subject, domain=domain,
                            entity_type=entityType, entity_ref=entityRef,
                            min_score=MIN_SCORE if minScore is None else minScore,
                            include_stale=includeStale)
    # Asked only when nothing current came back, so the answer can say "expired,
    # refetch" instead of "nothing known" — two different situations that used to
    # produce the same note.
    expired = ([] if matches else
               [m for m in _index.search(question, limit=limit, subject=subject,
                                         domain=domain, entity_type=entityType,
                                         entity_ref=entityRef,
                                         min_score=MIN_SCORE if minScore is None else minScore,
                                         include_stale=True) if m.get("is_stale")])
    return {
        "question": question,
        "indexed_claims": _index.size(),
        "matches": matches,
        "note": _note(question, matches, expired),
        "stale_held": len(expired),
        "min_score": MIN_SCORE if minScore is None else minScore,
        "usage": ("Claims already fetched from the web, each with its sources and "
                  "status. Same rules as when they were fetched: attribute them, and "
                  "never take price, availability or cancellation from here."),
    }
