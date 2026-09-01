"""Company facts, agency facts, the official-advisory gate, and the no-fetch rule.

The shared point of every test here: a field nobody sourced must come back as
not verified. An enrichment layer that quietly fills a gap from the model's own
recollection is worse than one that returns nothing, because the gap is then
indistinguishable from a fact.
"""
import pytest

import web_tools
from enrichment_index import EnrichmentIndex, SqliteVectorStore
from web_enrich import (
    AUTHORITATIVE, FIELD_SCHEMA, FRESH_FOR_SECONDS, VERIFIED_ONLY_FIELDS, Cache, Claim,
    Enricher, Enrichment, Source, canonical_field, is_government_source,
    source_tier, validate_claims,
)

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


class StubProvider:
    """Returns the claims it was given, for the domains it was told to handle."""

    def __init__(self, claims, domains=("company_facts",), name="stub"):
        self.name = name
        self._claims = claims
        self._domains = set(domains)
        self.calls = []

    def handles(self, domain):
        return domain in self._domains

    async def fetch(self, subject, domain, context):
        self.calls.append((subject, domain))
        return [Claim(**{**c, "domain": domain}) for c in self._claims]


def claim(field_name, value, url, authority="third_party"):
    return {"field_name": field_name, "value": value, "provider": "stub",
            "authority": authority,
            "sources": [Source(url=url, tier=source_tier(url))]}


# ---- who counts as a government ----

@pytest.mark.parametrize("url", [
    "https://www.gov.uk/foreign-travel-advice/saudi-arabia",
    "https://travel.state.gov/content/travel/en/traveladvisories.html",
    "https://www.smartraveller.gov.au/destinations/middle-east/saudi-arabia",
    "https://www.mofa.gov.sa/en/Pages/default.aspx",
    "https://travel.gc.ca/destinations/saudi-arabia",
    "https://www.canada.ca/en/services/travel.html",
    "https://www.mofa.go.jp/index.html",
    "https://www.diplomatie.gouv.fr/fr/",
    "https://www.who.int/travel-advice",
])
def test_a_government_page_is_recognised(url):
    assert is_government_source(url) is True
    assert source_tier(url) == "gov"


@pytest.mark.parametrize("url", [
    "https://www.reuters.com/world/uk-updates-travel-advice-2026-01-02/",
    "https://www.bbc.co.uk/news/world-12345678",
    "https://www.tripadvisor.com/ShowTopic-g294003",
    "https://someblog.wordpress.com/travel-warning",
    "https://www.booking.com/country/sa.html",
    # The reason the check reads the host structurally instead of searching for
    # "gov" anywhere in it. Every one of these contains a government string and
    # none of them is a government.
    "https://gov.uk.travel-deals.com/advice",
    "https://www.gov.uk.phishing.example/advice",
    "https://notgov.com/travel",
    "https://example.com/gov.uk/advice",
    "https://travel-state-gov.co/advisory",
])
def test_a_page_that_merely_looks_official_is_not(url):
    assert is_government_source(url) is False
    assert source_tier(url) != "gov"


# ---- the advisory may only be called official when a government said it ----

@pytest.mark.asyncio
async def test_a_news_report_of_an_advisory_is_not_an_official_advisory():
    provider = StubProvider(
        [claim("advisory_level", "avoid all but essential travel",
               "https://www.reuters.com/world/advice-2026/"),
         claim("guidance", "review your plans before travelling",
               "https://www.bbc.co.uk/news/world-1")],
        domains=("advisory",))
    result = await Enricher([provider], Cache()).enrich("Beirut", "advisory")
    model = result.to_model()

    assert model["checks"]["official_advisory_verified"] is False
    assert "no official advisory was verified" in model["note"]
    # The claims are still reported — attributed to the sites that published
    # them. Withholding them would be its own kind of dishonesty.
    assert set(model["findings"]) == {"advisory_level", "guidance"}
    assert all(entry["authority"] == "third_party"
               for entries in model["findings"].values() for entry in entries)
    assert all(entry["verified"] is False
               for entries in model["findings"].values() for entry in entries)


@pytest.mark.asyncio
async def test_a_government_advisory_is_official():
    provider = StubProvider(
        [claim("advisory_level", "avoid all but essential travel to parts",
               "https://www.gov.uk/foreign-travel-advice/lebanon")],
        domains=("advisory",))
    model = (await Enricher([provider], Cache()).enrich("Beirut", "advisory")).to_model()

    assert model["checks"]["official_advisory_verified"] is True
    assert model["checks"]["official_sources"] == [
        "https://www.gov.uk/foreign-travel-advice/lebanon"]
    assert model["findings"]["advisory_level"][0]["authority"] == "government"
    assert model["findings"]["advisory_level"][0]["verified"] is True


@pytest.mark.asyncio
async def test_one_government_source_among_news_reports_is_enough():
    provider = StubProvider(
        [claim("advisory_level", "exercise caution",
               "https://www.aljazeera.com/news/2026/1/1/travel"),
         claim("guidance", "check entry rules before you fly",
               "https://www.gov.uk/foreign-travel-advice/jordan")],
        domains=("advisory",))
    model = (await Enricher([provider], Cache()).enrich("Amman", "advisory")).to_model()

    assert model["checks"]["official_advisory_verified"] is True
    assert model["checks"]["unofficial_claims"] == ["advisory_level"]


# ---- company facts ----

@pytest.mark.asyncio
async def test_company_facts_carry_the_required_fields_with_a_source_each():
    provider = StubProvider([
        claim("legal_name", "Hilton Worldwide Holdings Inc.",
              "https://ir.hilton.com/", authority="reference_backed"),
        claim("parent_company_or_owner", "Hilton Worldwide Holdings Inc.",
              "https://ir.hilton.com/", authority="reference_backed"),
        claim("headquarters", "McLean, Virginia", "https://www.hilton.com/en/corporate/"),
        claim("founded", "1919", "https://www.hilton.com/en/corporate/"),
        claim("official_website", "https://www.hilton.com",
              "https://www.hilton.com", authority="entity_official"),
    ])
    result = await Enricher([provider], Cache()).enrich(
        "Hilton Worldwide", "company_facts", entity_type="company",
        entity_ref="Hilton Worldwide")
    model = result.to_model()

    for wanted in ("legal_name", "parent_company_or_owner", "headquarters",
                   "founded", "official_website"):
        assert wanted in model["findings"], wanted
        entry = model["findings"][wanted][0]
        assert entry["sources"] and entry["sources"][0]["url"].startswith("http")
        assert entry["observed_at"] and entry["valid_until"]
    assert model["entity_type"] == "company"
    assert model["checks"]["schema"] == list(FIELD_SCHEMA["company_facts"])


@pytest.mark.asyncio
async def test_a_ceo_no_authoritative_source_stands_behind_is_not_reported():
    """The whole point of the verified-only rule. A travel blog naming a chief
    executive is a rumour, and the answer has to be able to say so."""
    provider = StubProvider([
        claim("legal_name", "Acme Hotels Ltd", "https://acmehotels.example.com/about",
              authority="entity_official"),
        claim("ceo", "Someone Plausible", "https://travelgossip.example.com/who-runs-what"),
    ])
    model = (await Enricher([provider], Cache()).enrich(
        "Acme Hotels", "company_facts", entity_type="company")).to_model()

    assert "ceo" not in model["findings"]
    assert "ceo" in model["checks"]["not_verified"]
    assert "not verified" in model["note"]


@pytest.mark.asyncio
async def test_a_ceo_the_companys_own_site_names_is_reported():
    provider = StubProvider([
        claim("official_website", "https://acmehotels.example.com",
              "https://acmehotels.example.com", authority="entity_official"),
        # Only the url says this is authoritative — the stub claims nothing.
        claim("ceo", "Real Person", "https://acmehotels.example.com/leadership"),
    ])
    model = (await Enricher([provider], Cache()).enrich(
        "Acme Hotels", "company_facts", entity_type="company")).to_model()

    assert model["findings"]["ceo"][0]["value"] == "Real Person"
    assert model["findings"]["ceo"][0]["authority"] == "entity_official"
    assert model["checks"]["not_verified"] == []


@pytest.mark.asyncio
async def test_a_field_the_schema_does_not_carry_is_dropped_not_passed_through():
    """A closed schema is what makes "we do not carry that" a true statement."""
    provider = StubProvider([
        claim("legal_name", "Acme Hotels Ltd", "https://acmehotels.example.com/about"),
        claim("annual_revenue", "$4.1bn", "https://acmehotels.example.com/investors"),
        claim("employee_count", "142,000", "https://acmehotels.example.com/careers"),
    ])
    model = (await Enricher([provider], Cache()).enrich(
        "Acme Hotels", "company_facts", entity_type="company")).to_model()

    assert "annual_revenue" not in model["findings"]
    assert "employee_count" not in model["findings"]
    assert model["checks"]["fields_outside_schema"] == ["annual_revenue", "employee_count"]


@pytest.mark.asyncio
async def test_what_was_not_found_is_named_rather_than_left_blank():
    provider = StubProvider([
        claim("legal_name", "Acme Hotels Ltd", "https://acmehotels.example.com/about")])
    model = (await Enricher([provider], Cache()).enrich(
        "Acme Hotels", "company_facts", entity_type="company")).to_model()

    assert model["checks"]["fields_present"] == ["legal_name"]
    assert set(model["checks"]["missing"]) == {
        "parent_company_or_owner", "headquarters", "founded", "official_website", "ceo"}
    assert "not found" in model["note"]


def test_source_names_are_mapped_to_the_schema_rather_than_lost():
    assert canonical_field("company_facts", "HQ") == "headquarters"
    assert canonical_field("company_facts", "Head Office") == "headquarters"
    assert canonical_field("company_facts", "founded_year") == "founded"
    assert canonical_field("company_facts", "Chief Executive Officer") == "ceo"
    assert canonical_field("company_facts", "website") == "official_website"
    assert canonical_field("company_facts", "market_cap") is None
    assert canonical_field("agency_facts", "IATA number") == "accreditation_or_licence"
    assert canonical_field("agency_facts", "telephone") == "contact_phone"
    # A domain without a closed schema keeps whatever it was given.
    assert canonical_field("weather", "forecast_2026-09-04") == "forecast_2026-09-04"


# ---- agency facts ----

@pytest.mark.asyncio
async def test_agency_facts_carry_the_required_fields():
    provider = StubProvider([
        claim("legal_or_trading_name", "Rihla Travel LLC",
              "https://rihla.example.com/about", authority="entity_official"),
        claim("country", "Saudi Arabia", "https://rihla.example.com/about"),
        claim("headquarters_or_address", "King Fahd Road, Riyadh",
              "https://rihla.example.com/contact"),
        claim("official_website", "https://rihla.example.com",
              "https://rihla.example.com", authority="entity_official"),
        claim("contact_phone", "+966 11 000 0000", "https://rihla.example.com/contact"),
        claim("contact_email", "hello@rihla.example.com",
              "https://rihla.example.com/contact"),
        claim("accreditation_or_licence", "IATA 12-3 4567 8",
              "https://rihla.example.com/legal"),
    ], domains=("agency_facts",))
    model = (await Enricher([provider], Cache()).enrich(
        "Rihla Travel", "agency_facts", entity_type="agency")).to_model()

    for wanted in FIELD_SCHEMA["agency_facts"]:
        assert wanted in model["findings"], wanted
        assert model["findings"][wanted][0]["sources"][0]["url"].startswith("http")
        assert model["findings"][wanted][0]["observed_at"]
    assert model["checks"]["missing"] == []


@pytest.mark.asyncio
async def test_contact_details_and_a_licence_from_a_directory_are_not_verified():
    """A licence number copied off an aggregator looks exactly like a real one.
    That is what makes reporting it unsourced worse than reporting nothing."""
    provider = StubProvider([
        claim("legal_or_trading_name", "Rihla Travel LLC",
              "https://www.some-directory.example.com/rihla"),
        claim("contact_phone", "+966 11 999 9999",
              "https://www.some-directory.example.com/rihla"),
        claim("contact_email", "info@not-really-them.example.com",
              "https://www.some-directory.example.com/rihla"),
        claim("accreditation_or_licence", "IATA 99-9 9999 9",
              "https://www.some-directory.example.com/rihla"),
    ], domains=("agency_facts",))
    model = (await Enricher([provider], Cache()).enrich(
        "Rihla Travel", "agency_facts", entity_type="agency")).to_model()

    assert "contact_phone" not in model["findings"]
    assert "contact_email" not in model["findings"]
    assert "accreditation_or_licence" not in model["findings"]
    assert set(model["checks"]["not_verified"]) == {
        "contact_phone", "contact_email", "accreditation_or_licence"}
    # The name is not on the verified-only list, so it survives with its source.
    assert model["findings"]["legal_or_trading_name"][0]["authority"] == "third_party"


@pytest.mark.asyncio
async def test_a_licence_confirmed_by_a_regulator_is_verified():
    provider = StubProvider([
        claim("accreditation_or_licence", "ATOL 1234",
              "https://www.caa.gov.uk/atol-holders/rihla-travel")],
        domains=("agency_facts",))
    model = (await Enricher([provider], Cache()).enrich(
        "Rihla Travel", "agency_facts", entity_type="agency")).to_model()

    assert model["findings"]["accreditation_or_licence"][0]["authority"] == "government"
    assert model["checks"]["not_verified"] == []


def test_the_verified_only_fields_are_the_ones_a_customer_would_act_on():
    assert VERIFIED_ONLY_FIELDS["company_facts"] == frozenset({"ceo"})
    assert VERIFIED_ONLY_FIELDS["agency_facts"] == frozenset(
        {"contact_phone", "contact_email", "accreditation_or_licence"})
    assert all(f in AUTHORITATIVE for f in
               ("government", "entity_official", "reference_backed"))
    assert "third_party" not in AUTHORITATIVE


# ---- freshness ----

def test_the_new_domains_have_a_freshness_window():
    assert FRESH_FOR_SECONDS["company_facts"] == 30 * 24 * 3600
    assert FRESH_FOR_SECONDS["agency_facts"] == 30 * 24 * 3600
    # Every domain must have one, or a claim is indexed without an expiry and
    # is then held for ever.
    from web_enrich import DOMAINS
    assert set(DOMAINS) <= set(FRESH_FOR_SECONDS)


def test_a_stored_company_claim_expires_on_its_own_window(monkeypatch):
    from datetime import datetime, timedelta, timezone
    index = EnrichmentIndex(SqliteVectorStore(":memory:"))
    old = datetime.now(timezone.utc) - timedelta(days=31)
    enrichment = Enrichment(subject="Acme Hotels", domain="company_facts",
                            entity_type="company", entity_ref="Acme Hotels")
    enrichment.claims = [Claim(domain="company_facts", field_name="headquarters",
                               value="McLean, Virginia", observed_at=old,
                               sources=[Source(url="https://acme.example.com/about")])]
    index.add(enrichment)

    assert index.search("where is Acme Hotels headquartered", limit=5) == []
    stale = index.search("where is Acme Hotels headquartered", limit=5, include_stale=True)
    assert stale and stale[0]["is_stale"] is True


# ---- fetch, index, then find it again in another session ----

def _fresh_index(monkeypatch):
    index = EnrichmentIndex(SqliteVectorStore(":memory:"))
    monkeypatch.setattr(web_tools, "_index", index)
    return index


@pytest.mark.asyncio
async def test_company_facts_are_indexed_and_found_again_by_a_later_session(monkeypatch):
    index = _fresh_index(monkeypatch)
    provider = StubProvider([
        claim("legal_name", "Hilton Worldwide Holdings Inc.", "https://ir.hilton.com/"),
        claim("headquarters", "McLean, Virginia", "https://ir.hilton.com/"),
        claim("founded", "1919", "https://ir.hilton.com/"),
        claim("parent_company_or_owner", "Hilton Worldwide Holdings Inc.",
              "https://ir.hilton.com/"),
    ])
    monkeypatch.setattr(web_tools, "_enricher",
                        Enricher([provider], Cache(), index=index))

    fetched = await web_tools.enrich_company_facts("Hilton Worldwide")
    assert fetched["domains"]["company_facts"]["findings"]["headquarters"]

    # A different session, asking in its own words, with no fetch of its own.
    found = await web_tools.search_enrichment("where is Hilton Worldwide headquartered")
    assert found["matches"], found["note"]
    assert {m["field"] for m in found["matches"]} & {"headquarters"}
    assert all(m["entity_type"] == "company" for m in found["matches"])
    assert found["matches"][0]["sources"][0]["url"].startswith("http")

    owner = await web_tools.search_enrichment("who owns Hilton Worldwide",
                                              domain="company_facts")
    assert {m["field"] for m in owner["matches"]} & {"parent_company_or_owner"}


@pytest.mark.asyncio
async def test_agency_facts_are_indexed_and_found_again_by_a_later_session(monkeypatch):
    index = _fresh_index(monkeypatch)
    provider = StubProvider([
        claim("legal_or_trading_name", "Rihla Travel LLC",
              "https://rihla.example.com/about", authority="entity_official"),
        claim("country", "Saudi Arabia", "https://rihla.example.com/about"),
        claim("headquarters_or_address", "King Fahd Road, Riyadh",
              "https://rihla.example.com/contact"),
    ], domains=("agency_facts",))
    monkeypatch.setattr(web_tools, "_enricher",
                        Enricher([provider], Cache(), index=index))

    await web_tools.enrich_agency_facts("Rihla Travel", country="Saudi Arabia")

    found = await web_tools.search_enrichment("what is the address of Rihla Travel")
    assert found["matches"], found["note"]
    assert all(m["entity_type"] == "agency" for m in found["matches"])
    assert {m["field"] for m in found["matches"]} & {"headquarters_or_address"}

    scoped = await web_tools.search_enrichment("Rihla Travel", entityType="agency",
                                               entityRef="Rihla Travel")
    assert scoped["matches"]


@pytest.mark.asyncio
async def test_the_index_keeps_a_company_and_an_agency_apart(monkeypatch):
    """Both domains are about organisations and share half their vocabulary.
    Retrieval has to keep them separate or one answers for the other."""
    index = _fresh_index(monkeypatch)
    monkeypatch.setattr(web_tools, "_enricher", Enricher(
        [StubProvider([claim("headquarters", "McLean, Virginia",
                             "https://ir.hilton.com/")]),
         StubProvider([claim("headquarters_or_address", "King Fahd Road, Riyadh",
                             "https://rihla.example.com/contact")],
                      domains=("agency_facts",), name="stub-agency")],
        Cache(), index=index))

    await web_tools.enrich_company_facts("Hilton Worldwide")
    await web_tools.enrich_agency_facts("Rihla Travel")

    # Asked as questions, not as bare names: a lone proper noun carries no
    # domain words and scores below the floor by design.
    company = await web_tools.search_enrichment("where is Hilton headquartered")
    agency = await web_tools.search_enrichment("what is the address of Rihla Travel")
    assert {m["entity_ref"] for m in company["matches"]} == {"Hilton Worldwide"}
    assert {m["entity_ref"] for m in agency["matches"]} == {"Rihla Travel"}
    assert {m["entity_type"] for m in company["matches"]} == {"company"}
    assert {m["entity_type"] for m in agency["matches"]} == {"agency"}


@pytest.mark.asyncio
async def test_asking_about_a_company_nobody_fetched_says_so(monkeypatch):
    _fresh_index(monkeypatch)
    found = await web_tools.search_enrichment("who owns Marriott International")
    assert found["matches"] == []
    assert "Marriott" in found["note"]
    assert "enrich_company_facts" in found["note"]


@pytest.mark.asyncio
async def test_the_new_entity_types_and_domains_are_accepted(monkeypatch):
    _fresh_index(monkeypatch)
    for entity_type in ("hotel", "city", "company", "agency"):
        await web_tools.search_enrichment("anything", entityType=entity_type)
    for domain in ("company_facts", "agency_facts"):
        await web_tools.search_enrichment("anything", domain=domain)
    with pytest.raises(ValueError):
        await web_tools.search_enrichment("anything", entityType="supplier")
    with pytest.raises(ValueError):
        await web_tools.search_enrichment("anything", domain="financials")


# ---- validation is a function, checkable on its own ----

def test_validation_reports_what_it_removed_and_why():
    claims = [Claim(domain="company_facts", field_name="HQ", value="McLean",
                    sources=[Source(url="https://acme.example.com/about")]),
              Claim(domain="company_facts", field_name="ceo", value="Someone",
                    sources=[Source(url="https://gossip.example.com/x")]),
              Claim(domain="company_facts", field_name="share_price", value="41",
                    sources=[Source(url="https://acme.example.com/ir")])]
    kept, checks = validate_claims("company_facts", claims, {})

    assert [c.field_name for c in kept] == ["headquarters"]
    assert checks["not_verified"] == ["ceo"]
    assert checks["fields_outside_schema"] == ["share_price"]
    assert "ceo" in checks["missing"]


def test_a_confirmed_official_website_makes_the_subjects_own_pages_authoritative():
    """Found and then not used is the failure this prevents: a fact quoted from
    the company's own leadership page ranking level with a directory."""
    claims = [Claim(domain="company_facts", field_name="official_website",
                    value="https://acme.example.com",
                    sources=[Source(url="https://acme.example.com")]),
              Claim(domain="company_facts", field_name="ceo", value="Real Person",
                    sources=[Source(url="https://acme.example.com/leadership")])]
    kept, checks = validate_claims("company_facts", claims, {})

    assert {c.field_name for c in kept} == {"official_website", "ceo"}
    assert checks["not_verified"] == []
