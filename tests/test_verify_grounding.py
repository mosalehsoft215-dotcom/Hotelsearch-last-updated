"""What the answer is allowed to say about company, agency and advisory facts.

The enrichment layer decides what comes back; these check what gets written.
Both halves are needed. A tool that correctly reports "no authoritative source
for ceo" achieves nothing if the answer then names one anyway, and the model has
every incentive to — it knows the answer, or thinks it does.
"""
import pytest

from agents.hotel_search_agent import HotelSearchAgent
from runtime import AgentContext, ToolCall

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def advisory_call(url, official):
    """An enrich_destination result carrying one advisory claim."""
    return ToolCall(
        name="enrich_destination", args={"city": "Beirut"},
        result={"city": "Beirut", "domains": {"advisory": {
            "domain": "advisory", "entity_type": "city", "entity_ref": "Beirut",
            "findings": {"advisory_level": [{
                "value": "avoid all but essential travel", "status": "single_source",
                "authority": "government" if official else "third_party",
                "verified": official,
                "sources": [{"url": url, "title": None,
                             "tier": "gov" if official else "news"}],
                "observed_at": "2026-09-01T00:00:00+00:00"}]},
            "checks": {"official_advisory_verified": official,
                       "official_sources": [url] if official else [],
                       "unofficial_claims": [] if official else ["advisory_level"]},
            "note": None if official else (
                "no official advisory was verified: nothing here was published by a "
                "government or recognised official authority")}}})


def stored_advisory_call(url):
    """The same claim reached through the index rather than a fresh fetch."""
    return ToolCall(
        name="search_enrichment", args={"question": "travel advice for Beirut"},
        result={"question": "travel advice for Beirut", "matches": [{
            "subject": "Beirut", "entity_type": "city", "entity_ref": "Beirut",
            "domain": "advisory", "field": "advisory_level",
            "value": "avoid all but essential travel", "status": "single_source",
            "sources": [{"url": url, "title": None, "tier": "other"}],
            "observed_at": "2026-09-01T00:00:00+00:00", "is_stale": False,
            "match": 0.81}], "note": None})


def company_call(present, not_verified, missing):
    return ToolCall(
        name="enrich_company_facts", args={"companyName": "Acme Hotels"},
        result={"company": "Acme Hotels", "domains": {"company_facts": {
            "domain": "company_facts", "entity_type": "company",
            "entity_ref": "Acme Hotels",
            "findings": {name: [{"value": value, "status": "single_source",
                                 "authority": "entity_official", "verified": True,
                                 "sources": [{"url": "https://acme.example.com/about",
                                              "title": None, "tier": "official"}],
                                 "observed_at": "2026-09-01T00:00:00+00:00"}]
                         for name, value in present.items()},
            "checks": {"schema": ["legal_name", "parent_company_or_owner",
                                  "headquarters", "founded", "official_website", "ceo"],
                       "fields_present": sorted(present),
                       "not_verified": not_verified, "missing": missing,
                       "fields_outside_schema": []},
            "note": "no authoritative source for " + ", ".join(not_verified)
                    if not_verified else None}}})


async def verdict(answer, *calls, stored_only=False):
    ctx = AgentContext(org_id=ORG)
    ctx.stored_only = stored_only
    ctx.tool_calls.extend(calls)
    ctx.remember("last_answer", answer)
    return await HotelSearchAgent().verify(ctx)


# ---- the word "official" is a claim about the publisher ----

@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    "The official government travel advisory says to avoid all but essential travel.",
    "According to the official travel advisory, avoid non-essential travel.",
    "The government advisory for Beirut is to avoid all but essential travel.",
    "Official guidance is to avoid all but essential travel to Beirut.",
])
async def test_calling_a_news_report_an_official_advisory_fails(answer):
    result = await verdict(answer, advisory_call(
        "https://www.reuters.com/world/lebanon-travel-2026/", official=False))

    assert result.passed is False
    assert any("official government travel advisory" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_the_same_wording_passes_when_a_government_published_it():
    result = await verdict(
        "The official government travel advisory says to avoid all but essential travel.",
        advisory_call("https://www.gov.uk/foreign-travel-advice/lebanon", official=True))

    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_a_stored_government_claim_also_satisfies_the_check():
    """A claim retrieved from the index is checked by its source, the same way
    it was when it was fetched."""
    result = await verdict(
        "The official government travel advisory is to avoid all but essential travel.",
        stored_advisory_call("https://www.gov.uk/foreign-travel-advice/lebanon"))
    assert result.passed is True, result.issues

    leaked = await verdict(
        "The official government travel advisory is to avoid all but essential travel.",
        stored_advisory_call("https://www.aljazeera.com/news/2026/1/1/lebanon"))
    assert leaked.passed is False


@pytest.mark.asyncio
async def test_reporting_that_no_official_advisory_was_found_is_not_a_failure():
    """The honest answer must not be the one that fails. Same trap as the
    confirmed-price check: this sentence contains "official advisory"."""
    result = await verdict(
        "No official advisory was verified for Beirut. Reuters reports that the "
        "Foreign Office advises against all but essential travel; that is a news "
        "report, not the advisory itself.",
        advisory_call("https://www.reuters.com/world/lebanon-travel-2026/",
                      official=False))

    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_attributing_a_report_to_the_site_that_published_it_passes():
    """Naming the government body while crediting the outlet is the correct
    form, and must not be what fails."""
    result = await verdict(
        "Reuters reports that the Foreign Office advises against all but essential "
        "travel to Beirut. No government source was found, so this is not an "
        "official advisory.",
        advisory_call("https://www.reuters.com/world/lebanon-travel-2026/",
                      official=False))

    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_an_answer_with_no_advisory_wording_is_not_examined():
    result = await verdict(
        "Beirut is on the Mediterranean coast, and the airport is close to downtown.",
        advisory_call("https://www.reuters.com/world/lebanon-travel-2026/",
                      official=False))
    assert result.passed is True, result.issues


# ---- a field nobody sourced may not reappear with a value ----

@pytest.mark.asyncio
async def test_naming_a_ceo_that_came_back_unverified_fails():
    result = await verdict(
        "Acme Hotels Ltd is headquartered in McLean. The CEO is Christopher Nassetta.",
        company_call({"legal_name": "Acme Hotels Ltd", "headquarters": "McLean"},
                     not_verified=["ceo"], missing=[]))

    assert result.passed is False
    assert any("ceo" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_saying_the_ceo_was_not_verified_passes():
    result = await verdict(
        "Acme Hotels Ltd is headquartered in McLean. The chief executive is not "
        "verified — no authoritative source named one.",
        company_call({"legal_name": "Acme Hotels Ltd", "headquarters": "McLean"},
                     not_verified=["ceo"], missing=[]))

    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_supplying_a_founding_year_nothing_returned_fails():
    result = await verdict(
        "Acme Hotels Ltd was founded in 1919 and is headquartered in McLean.",
        company_call({"headquarters": "McLean"},
                     not_verified=[], missing=["founded", "legal_name"]))

    assert result.passed is False
    assert any("founded" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_a_ceo_that_was_returned_may_be_named():
    result = await verdict(
        "Acme Hotels Ltd is headquartered in McLean. The CEO is Real Person.",
        company_call({"legal_name": "Acme Hotels Ltd", "headquarters": "McLean",
                      "ceo": "Real Person"},
                     not_verified=[], missing=[]))

    assert result.passed is True, result.issues


@pytest.mark.asyncio
async def test_quoting_an_unverified_licence_number_fails():
    call = ToolCall(
        name="enrich_agency_facts", args={"agencyName": "Rihla Travel"},
        result={"agency": "Rihla Travel", "domains": {"agency_facts": {
            "domain": "agency_facts", "entity_type": "agency",
            "entity_ref": "Rihla Travel",
            "findings": {"legal_or_trading_name": [{
                "value": "Rihla Travel LLC", "status": "single_source",
                "authority": "third_party", "verified": False,
                "sources": [{"url": "https://directory.example.com/rihla",
                             "title": None, "tier": "other"}],
                "observed_at": "2026-09-01T00:00:00+00:00"}]},
            "checks": {"schema": [], "fields_present": ["legal_or_trading_name"],
                       "not_verified": ["accreditation_or_licence", "contact_email"],
                       "missing": [], "fields_outside_schema": []},
            "note": "no authoritative source for accreditation_or_licence"}}})

    bad = await verdict("Rihla Travel LLC holds IATA 12-3 4567 8 and can be reached "
                        "at info@rihla.example.com.", call)
    assert bad.passed is False
    assert any("accreditation_or_licence" in issue for issue in bad.issues)

    good = await verdict("Rihla Travel LLC is listed in a directory. Its licence "
                         "number and email address are not verified — no "
                         "authoritative source carried them.", call)
    assert good.passed is True, good.issues


@pytest.mark.asyncio
async def test_a_field_present_in_the_findings_is_never_flagged():
    """The check only ever looks at fields the fetch said it could not confirm."""
    result = await verdict(
        "Acme Hotels Ltd was founded in 1919, and its site is https://acme.example.com.",
        company_call({"founded": "1919", "official_website": "https://acme.example.com"},
                     not_verified=[], missing=["ceo"]))

    assert result.passed is True, result.issues
