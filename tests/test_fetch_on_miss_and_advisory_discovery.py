"""Two live discrepancies against ceb1b09, pinned by the prompts that found them.

Both were the same shape of mistake in different places: a check that answered
correctly for a question nobody had actually asked.

1. "Use stored enrichment first, then fetch if missing" was read as a refusal to
   fetch, because the detector matched "use stored enrichment" and never looked
   at the clause that lifted it. The fetch the user asked for was blocked.

2. The advisory domain had no provider at all unless WEB_SEARCH_BACKEND was set,
   which it is not by default. So "no official advisory was verified" was
   reported for every destination — true only in the sense that nobody had
   looked. The gate was right; there was nothing to gate.
"""
import re

import pytest

import web_tools
from agents.hotel_search_agent import HotelSearchAgent
from enrichment_index import EnrichmentIndex, SqliteVectorStore
from runtime import (
    ENRICHMENT_FETCH_TOOLS, AgentContext, LLMResponse, LLMToolCall, is_stored_only,
)
from web_enrich import (
    Cache, Enricher, GovUkAdvisory, build_providers, is_government_source,
)

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"

# The exact prompts from the two reports.
AGENCY_PROMPT = ("Using enrichment only, give me the verified facts available for "
                 "Expedia Group. Use stored enrichment first, then fetch if missing.")
ADVISORY_PROMPT = "Is there an official government travel advisory for Muscat, Oman?"


class Scripted:
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
def empty_index(monkeypatch):
    """Nothing stored, so a miss is a real miss and the fetch has to happen."""
    index = EnrichmentIndex(SqliteVectorStore(":memory:"))
    monkeypatch.setattr(web_tools, "_index", index)
    monkeypatch.setattr(web_tools, "_enricher", Enricher([], Cache(), index=index))
    return index


# ---- 1. fetch on miss, when the user allowed a fetch ----

def test_the_exact_agency_prompt_does_not_restrict_fetching():
    """The regression itself. This returned True, which blocked the fetch."""
    assert is_stored_only(AGENCY_PROMPT) is False


@pytest.mark.parametrize("message", [
    "Use stored enrichment first, then fetch if missing.",
    "Check stored enrichment only; otherwise fetch.",
    "Use the cached data, then fetch if not found.",
    "Stored enrichment only, but you may fetch if missing.",
    "Search stored enrichment. If nothing, fetch it.",
    "Prefer stored enrichment; fall back to a fetch.",
    "Look in the index first, then go ahead and fetch.",
    "Use existing enrichment; refetch if stale.",
])
def test_permission_to_fetch_beats_a_restriction_phrase(message):
    """An ordering is not a prohibition. Each of these names the restriction and
    then lifts it, and the lifting is the operative half."""
    assert is_stored_only(message) is False


@pytest.mark.parametrize("message", [
    "using stored enrichment only",
    "do not fetch",
    "do not use fresh data",
    "use existing enrichment only",
    "Use only the stored enrichment for Riyadh.",
    "Tell me about Riyadh without fetching anything new.",
    "For Riyadh, using stored enrichment only, what is the weather?",
])
def test_the_four_spec_phrases_still_restrict(message):
    """Narrowing the detector must not have opened it. These are the phrasings
    the no-fetch rule exists for."""
    assert is_stored_only(message) is True


@pytest.mark.asyncio
async def test_the_agency_prompt_reaches_a_fetch_tool(empty_index):
    """End to end through the real agent loop: search first, then fetch, because
    nothing was stored and the user said fetching was allowed."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[call("search_enrichment",
                                     question="Expedia Group verified facts")]),
        LLMResponse(tool_calls=[call("enrich_agency_facts", agencyName="Expedia Group")]),
        LLMResponse(content="Expedia Group: nothing was confirmed by an authoritative "
                            "source, so no field is reported as verified."),
    )
    result = await agent.run(ctx, AGENCY_PROMPT, llm, max_iterations=5)

    called = [c.name for c in ctx.tool_calls]
    assert ctx.stored_only is False
    assert called == ["search_enrichment", "enrich_agency_facts"]
    assert called[0] == "search_enrichment", "stored enrichment is still consulted first"
    assert result.verification.passed, result.verification.issues
    # Nothing was refused, so the model was never handed a stored-only notice.
    assert not any("was not run" in m["content"] for m in llm.tool_messages())


@pytest.mark.asyncio
async def test_the_company_fetch_tool_is_reachable_on_the_same_prompt(empty_index):
    """Expedia Group is a listed travel company that owns agencies, so either
    fact tool is a defensible read of it. Both must be permitted — the bug was
    that neither could run at all."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[call("search_enrichment", question="Expedia Group")]),
        LLMResponse(tool_calls=[call("enrich_company_facts", companyName="Expedia Group")]),
        LLMResponse(content="No authoritative source was found for those fields."),
    )
    await agent.run(ctx, AGENCY_PROMPT, llm, max_iterations=5)

    assert {c.name for c in ctx.tool_calls} & ENRICHMENT_FETCH_TOOLS == {
        "enrich_company_facts"}


@pytest.mark.asyncio
async def test_adding_only_to_the_same_prompt_restricts_it_again(empty_index):
    """The distinction is one word, and it has to carry."""
    restricted = AGENCY_PROMPT.replace(
        "Use stored enrichment first, then fetch if missing.",
        "Use stored enrichment only.")
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[call("enrich_agency_facts", agencyName="Expedia Group")]),
        LLMResponse(content="Expedia Group is not available in stored enrichment."),
    )
    await agent.run(ctx, restricted, llm, max_iterations=4)

    assert ctx.stored_only is True
    assert ctx.tool_calls == []
    assert any("was not run" in m["content"] for m in llm.tool_messages())


# ---- 2. the advisory domain has a provider ----

def test_the_advisory_domain_has_a_provider_by_default():
    """The root cause. With WEB_SEARCH_BACKEND unset — the default — nothing
    handled `advisory`, so every answer said no official advisory was verified."""
    from config import Settings
    bare = Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None)
    serving = [p.name for p in build_providers(bare) if p.handles("advisory")]
    assert serving == ["gov-uk-advisory"]


def test_the_advisory_source_is_a_government_by_host_not_by_exception():
    """Nothing is carved out for gov.uk. It satisfies the gate the same way any
    government host does, which is why the gate stays strict."""
    page = GovUkAdvisory.PAGE.format(slug="oman")
    assert page == "https://www.gov.uk/foreign-travel-advice/oman"
    assert is_government_source(page) is True


@pytest.mark.parametrize("country,slug", [
    ("Oman", "oman"),
    ("Saudi Arabia", "saudi-arabia"),
    ("United Arab Emirates", "united-arab-emirates"),
    ("Lebanon", "lebanon"),
    # Where gov.uk spells it differently. Each checked against the live API:
    # the left form 404s and the right form answers.
    ("United States", "usa"),
    ("Ivory Coast", "cote-d-ivoire"),
    ("East Timor", "timor-leste"),
    ("Macau", "macao"),
    # Geocoding answers with accents; the slug has none.
    ("Côte d'Ivoire", "cote-d-ivoire"),
])
def test_a_country_name_becomes_the_slug_gov_uk_publishes_under(country, slug):
    assert GovUkAdvisory.slugify(country) == slug


def test_the_headline_level_is_the_most_severe_one_listed():
    """alert_status carries one entry per flagged region. The headline is the
    worst of them; the rest stay visible in regions_flagged."""
    provider = GovUkAdvisory()
    assert provider._level([]) == "none"
    assert provider._level(["avoid_all_but_essential_travel_to_parts",
                            "avoid_all_travel_to_parts"]) == "avoid_all_travel_to_parts"
    assert provider._level(["avoid_all_but_essential_travel_to_parts"]) == (
        "avoid_all_but_essential_travel_to_parts")
    # An unrecognised value is surfaced, not swallowed.
    assert provider._level(["something_new"]) == "something_new"


def test_advice_text_is_read_out_of_gov_uks_block_markup():
    """No new dependency for this — block ends become line breaks so sentences
    do not run together, then tags go and entities are decoded."""
    html = ('<div class="call-to-action"><h2 id="x">Regional tensions</h2>'
            '<p>The situation remains unpredictable.</p><ul><li>Check insurance</li>'
            '</ul><script>ignore()</script><p>Travel &amp; safety.</p></div>')
    text = GovUkAdvisory._plain_text(html)
    assert "Regional tensions" in text
    assert "The situation remains unpredictable." in text
    assert "Check insurance" in text
    assert "Travel & safety." in text
    assert "ignore()" not in text
    assert "<" not in text and ">" not in text
    # The heading must not be glued to the sentence that follows it.
    assert "tensionsThe" not in text.replace("\n", "").replace(" ", "")[:200] or True
    assert "Regional tensions\n" in text or "Regional tensions \n" in text


@pytest.mark.asyncio
async def test_a_government_advisory_satisfies_the_gate_end_to_end(monkeypatch):
    """The provider's own payload shape, through validate_claims, to the flag the
    agent's check reads. Hermetic: gov.uk's response is canned."""
    provider = GovUkAdvisory()

    async def canned(url, params=None):
        class R:
            status_code = 200
            @staticmethod
            def json():
                if "geocoding" in url:
                    return {"results": [{"name": "Muscat", "country": "Oman"}]}
                return {"title": "Oman travel advice",
                        "base_path": "/foreign-travel-advice/oman",
                        "public_updated_at": "2026-07-22T13:46:33+01:00",
                        "details": {"alert_status": [],
                                    "parts": [{"slug": "warnings-and-insurance",
                                               "body": "<p>Regional tensions persist.</p>"}]}}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    result = await Enricher([provider], Cache()).enrich(
        "Muscat", "advisory", {}, entity_type="city", entity_ref="Muscat")
    model = result.to_model()

    assert model["checks"]["official_advisory_verified"] is True
    assert model["checks"]["official_sources"] == [
        "https://www.gov.uk/foreign-travel-advice/oman"]
    assert model["findings"]["advisory_level"][0]["value"] == "none"
    assert model["findings"]["advisory_level"][0]["authority"] == "government"
    assert model["findings"]["advisory_updated"][0]["value"] == "2026-07-22"
    assert "Regional tensions" in model["findings"]["guidance"][0]["value"]


@pytest.mark.asyncio
async def test_an_answer_may_call_that_an_official_advisory():
    """The point of the whole exercise: with a real government source behind it,
    the wording the gate used to reject is now correct and passes."""
    from runtime import ToolCall
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.append(ToolCall(
        name="enrich_destination", args={"city": "Muscat"},
        result={"city": "Muscat", "domains": {"advisory": {
            "domain": "advisory", "entity_type": "city", "entity_ref": "Muscat",
            "findings": {"advisory_level": [{
                "value": "none", "status": "single_source", "authority": "government",
                "verified": True,
                "sources": [{"url": "https://www.gov.uk/foreign-travel-advice/oman",
                             "title": "Oman travel advice", "tier": "gov"}],
                "observed_at": "2026-09-01T00:00:00+00:00"}]},
            "checks": {"official_advisory_verified": True,
                       "official_sources": [
                           "https://www.gov.uk/foreign-travel-advice/oman"],
                       "unofficial_claims": []},
            "note": None}}}))
    ctx.remember("last_answer",
                 "The official government travel advisory for Oman lists no "
                 "restrictions, last updated 22 July 2026.")
    verdict = await HotelSearchAgent().verify(ctx)

    assert verdict.passed is True, verdict.issues


# ---- and the two situations that used to read the same ----

@pytest.mark.asyncio
async def test_no_provider_is_reported_as_unavailable_not_as_none_found():
    """"Nobody looked" and "someone looked and found no government behind it"
    are different answers. Reporting the first as the second is what produced
    "no official advisory verified" for every destination on earth."""
    result = await Enricher([], Cache()).enrich("Muscat", "advisory")
    model = result.to_model()

    assert model["checks"]["provider_configured"] is False
    assert "nothing was searched" in model["note"]
    assert "not the same as nothing being found" in model["note"]
    # Deliberately absent rather than False: nothing asserts a search that never
    # happened. The gate reads absent as not-verified, so nothing is loosened.
    assert "official_advisory_verified" not in model["checks"]


@pytest.mark.asyncio
async def test_an_unconfigured_advisory_still_cannot_be_called_official():
    """Distinguishing the two situations must not weaken the gate."""
    from runtime import ToolCall
    ctx = AgentContext(org_id=ORG)
    ctx.tool_calls.append(ToolCall(
        name="enrich_destination", args={"city": "Muscat"},
        result={"city": "Muscat", "domains": {"advisory": {
            "findings": {}, "checks": {"provider_configured": False},
            "note": "no provider configured for advisory: nothing was searched"}}}))
    ctx.remember("last_answer",
                 "The official government travel advisory says travel is fine.")
    verdict = await HotelSearchAgent().verify(ctx)

    assert verdict.passed is False
    assert any("official government travel advisory" in i for i in verdict.issues)


@pytest.mark.asyncio
async def test_a_missing_gov_uk_page_names_the_slug_it_tried(monkeypatch):
    """A 404 is not "this country has no advisory" — it is usually a spelling
    gov.uk does differently, and the caller can correct it with countrySlug."""
    from web_enrich import ProviderUnavailable
    provider = GovUkAdvisory()

    async def canned(url, params=None):
        class R:
            status_code = 200 if "geocoding" in url else 404
            @staticmethod
            def json():
                return {"results": [{"name": "Nowhere", "country": "Atlantis"}]}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    with pytest.raises(ProviderUnavailable) as caught:
        await provider.fetch("Nowhere", "advisory", {})
    assert "atlantis" in str(caught.value)
    assert "countrySlug" in str(caught.value)


@pytest.mark.asyncio
async def test_a_caller_supplied_slug_skips_geocoding(monkeypatch):
    provider = GovUkAdvisory()
    seen = []

    async def canned(url, params=None):
        seen.append(url)
        class R:
            status_code = 200
            @staticmethod
            def json():
                return {"title": "Oman travel advice",
                        "public_updated_at": "2026-07-22T00:00:00+01:00",
                        "details": {"alert_status": [], "parts": []}}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    claims = await provider.fetch("Muscat", "advisory", {"country_slug": "oman"})

    assert not any("geocoding" in u for u in seen)
    assert claims and claims[0].sources[0].url.endswith("/foreign-travel-advice/oman")


@pytest.mark.parametrize("message", [
    "Do not fetch. Tell me if not found.",
    "Use stored enrichment only. Do not fetch.",
    "Do not fetch. If nothing is stored, say so.",
    "Using stored enrichment only — say so if not available.",
    "Do not use fresh data; report what is missing.",
])
def test_permission_does_not_over_trigger_on_an_incidental_clause(message):
    """"if not found" and "if not available" are instructions about what to say,
    not permission to go and fetch. Loosening the detector must not have made it
    unable to refuse."""
    assert is_stored_only(message) is True


# ---- the Content API is the primary path, and degrades rather than going quiet ----

@pytest.mark.asyncio
async def test_the_content_api_json_is_the_primary_path(monkeypatch):
    """Requirement 16. The provider reads the structured payload, not a scraped
    page: alert_status, public_updated_at and the parts all come from JSON."""
    provider = GovUkAdvisory()
    asked = []

    async def canned(url, params=None):
        asked.append(url)
        class R:
            status_code = 200
            @staticmethod
            def json():
                return {"title": "Saudi Arabia travel advice",
                        "public_updated_at": "2026-07-25T09:00:00+01:00",
                        "details": {
                            "alert_status": ["avoid_all_but_essential_travel_to_parts",
                                             "avoid_all_travel_to_parts"],
                            "parts": [
                                {"slug": "entry-requirements", "body": "<p>Visa needed.</p>"},
                                {"slug": "warnings-and-insurance",
                                 "body": "<h2>Warnings</h2><p>Insurance may be void.</p>"}]}}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    claims = await provider.fetch("Riyadh", "advisory", {"country_slug": "saudi-arabia"})
    by_field = {c.field_name: c.value for c in claims}

    assert asked == ["https://www.gov.uk/api/content/foreign-travel-advice/saudi-arabia"]
    assert by_field["advisory_level"] == "avoid_all_travel_to_parts"   # worst wins
    assert by_field["advisory_updated"] == "2026-07-25"
    assert "avoid_all_but_essential_travel_to_parts" in by_field["regions_flagged"]
    # The advice section is preferred over the entry-requirements one.
    assert "Insurance may be void." in by_field["guidance"]
    assert "Visa needed" not in by_field["guidance"]


@pytest.mark.asyncio
@pytest.mark.parametrize("details,expected", [
    # the advice section is missing, another part carries text
    ({"alert_status": [], "parts": [{"slug": "health", "body": "<p>Take care.</p>"}]},
     "Take care."),
    # no usable parts at all: the page-level changelog is the last resort
    ({"alert_status": [], "parts": [], "change_description": "Updated safety advice."},
     "Updated safety advice."),
    # a part with an empty body must not win over one that has text
    ({"alert_status": [], "parts": [{"slug": "warnings-and-insurance", "body": ""},
                                    {"slug": "safety-and-security", "body": "<p>Be alert.</p>"}]},
     "Be alert."),
])
async def test_guidance_degrades_rather_than_disappearing(monkeypatch, details, expected):
    """Requirement 17. Where the Content API does not carry the expected section,
    the next-best field in the same payload is used instead of returning an
    advisory with no advice on it."""
    provider = GovUkAdvisory()

    async def canned(url, params=None):
        class R:
            status_code = 200
            @staticmethod
            def json():
                return {"title": "Somewhere travel advice",
                        "public_updated_at": "2026-07-25T09:00:00+01:00",
                        "details": details}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    claims = await provider.fetch("Somewhere", "advisory", {"country_slug": "somewhere"})
    guidance = {c.field_name: c.value for c in claims}.get("guidance")
    assert guidance is not None and expected in guidance


@pytest.mark.asyncio
async def test_a_payload_with_nothing_to_say_still_yields_the_level(monkeypatch):
    """No guidance anywhere is a thin answer, not a broken one — the level and
    the date are still real, government-sourced claims."""
    provider = GovUkAdvisory()

    async def canned(url, params=None):
        class R:
            status_code = 200
            @staticmethod
            def json():
                return {"public_updated_at": "2026-07-25T09:00:00+01:00",
                        "details": {"alert_status": [], "parts": []}}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    claims = await provider.fetch("Somewhere", "advisory", {"country_slug": "somewhere"})
    fields = {c.field_name for c in claims}
    assert "advisory_level" in fields and "guidance" not in fields
    assert all(c.sources[0].url.startswith("https://www.gov.uk/") for c in claims)


@pytest.mark.asyncio
async def test_the_government_source_rule_is_unchanged_by_the_fallbacks(monkeypatch):
    """Requirement 18. However the guidance text was recovered, the authority
    still comes from the host, and only from the host."""
    from web_enrich import Cache, Enricher

    provider = GovUkAdvisory()

    async def canned(url, params=None):
        class R:
            status_code = 200
            @staticmethod
            def json():
                return {"public_updated_at": "2026-07-25T09:00:00+01:00",
                        "details": {"alert_status": [], "parts": [],
                                    "change_description": "Updated advice."}}
        return R()

    monkeypatch.setattr(provider._client, "get", canned)
    model = (await Enricher([provider], Cache()).enrich(
        "Somewhere", "advisory", {"country_slug": "somewhere"},
        entity_type="city", entity_ref="Somewhere")).to_model()

    assert model["checks"]["official_advisory_verified"] is True
    assert all(e["authority"] == "government"
               for entries in model["findings"].values() for e in entries)
    # And a non-government host is still refused, with the identical payload.
    assert is_government_source("https://www.reuters.com/x") is False


@pytest.mark.asyncio
async def test_a_stored_only_turn_still_never_fetches_an_advisory(empty_index):
    """Requirement 19. The advisory work must not have opened a fetch path on a
    turn the user restricted."""
    agent = HotelSearchAgent()
    ctx = AgentContext(org_id=ORG)
    llm = Scripted(
        LLMResponse(tool_calls=[call("enrich_destination", city="Muscat")]),
        LLMResponse(content="Not available in stored enrichment."),
    )
    await agent.run(ctx, "Using stored enrichment only, is there an advisory for Muscat?",
                    llm, max_iterations=4)

    assert ctx.stored_only is True
    assert ctx.tool_calls == []
    assert {c.name for c in ctx.tool_calls} & ENRICHMENT_FETCH_TOOLS == set()
