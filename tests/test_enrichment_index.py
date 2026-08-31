"""Free-text search across what enrichment already fetched."""
from datetime import datetime, timedelta, timezone

import pytest

from enrichment_index import EnrichmentIndex, IndexedClaim, SqliteVectorStore, cosine
from graphiti_embedder import embed_text
from web_enrich import Cache, Claim, Enrichment, Enricher, ProviderUnavailable, Source


def claim(field_name, value, url="https://booking.com/a", domain="facilities"):
    return Claim(domain=domain, field_name=field_name, value=value,
                 sources=[Source(url=url, tier="reviews")])


def enrichment(subject, claims, domain="facilities"):
    return Enrichment(subject=subject, domain=domain, claims=claims)


def test_a_claim_embeds_its_subject_and_field_not_just_the_value():
    text = IndexedClaim(subject="Carawan Hotel, Jeddah", entity_type="hotel",
                        entity_ref="Carawan Hotel", domain="facilities",
                        field_name="pool", value="open all year", status="single_source",
                        sources=[], observed_at="2026-09-01T00:00:00+00:00").text
    assert "Carawan Hotel, Jeddah" in text and "pool" in text and "open all year" in text


def test_indexing_skips_claims_with_no_source():
    index = EnrichmentIndex(SqliteVectorStore())
    written = index.add(enrichment("Carawan Hotel, Jeddah", [
        claim("pool", "open all year"),
        Claim(domain="facilities", field_name="gym", value="present", sources=[])]))
    assert written == 1 and index.size() == 1


def test_a_question_finds_the_claim_without_naming_the_subject():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(enrichment("Carawan Hotel, Jeddah", [claim("pool", "rooftop pool open all year")]))
    index.add(enrichment("Central Inn, Makkah", [claim("shuttle", "free airport shuttle")]))
    hits = index.search("which hotel has a pool")
    assert hits and hits[0]["field"] == "pool"
    assert hits[0]["sources"][0]["url"] == "https://booking.com/a"
    assert hits[0]["match"] > 0


def test_results_can_be_narrowed_by_subject_and_domain():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(enrichment("Carawan Hotel, Jeddah", [claim("pool", "rooftop pool")]))
    index.add(enrichment("Central Inn, Makkah", [claim("pool", "indoor pool")]))
    index.add(enrichment("Jeddah", [claim("forecast", "hot and dry", domain="weather")],
                         domain="weather"))
    assert all("Central" in h["subject"] for h in index.search("pool", subject="Central"))
    assert all(h["domain"] == "weather" for h in index.search("pool", domain="weather"))


def test_re_fetching_updates_the_row_rather_than_adding_another():
    index = EnrichmentIndex(SqliteVectorStore())
    subject = "Carawan Hotel, Jeddah"
    index.add(enrichment(subject, [claim("pool", "open all year")]))
    index.add(enrichment(subject, [claim("pool", "closed for refurbishment")]))
    assert index.size() == 1
    assert "closed" in index.search("pool")[0]["value"]


def test_an_empty_question_returns_nothing():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(enrichment("Carawan Hotel, Jeddah", [claim("pool", "open")]))
    assert index.search("   ") == []


def test_the_index_survives_a_restart(tmp_path):
    path = tmp_path / "index.sqlite3"
    first = SqliteVectorStore(path)
    EnrichmentIndex(first).add(enrichment("Carawan Hotel, Jeddah", [claim("pool", "rooftop pool")]))
    first.close()
    reopened = EnrichmentIndex(SqliteVectorStore(path))
    assert reopened.size() == 1
    assert reopened.search("pool")[0]["field"] == "pool"


def test_identical_text_scores_higher_than_unrelated_text():
    a, b = embed_text("rooftop pool open all year"), embed_text("rooftop pool open all year")
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-9)
    assert cosine(a, embed_text("failed booking queue message")) < 0.5


# ---- the write-time hook ----

class Stub:
    name = "stub"

    def __init__(self, claims=None, fail=None):
        self._claims, self._fail = claims or [], fail

    def handles(self, domain):
        return True

    async def fetch(self, subject, domain, context):
        if self._fail:
            raise ProviderUnavailable(self._fail)
        return self._claims


@pytest.mark.asyncio
async def test_claims_are_indexed_in_the_same_call_that_fetches_them():
    index = EnrichmentIndex(SqliteVectorStore())
    enricher = Enricher([Stub([claim("pool", "rooftop pool open all year")])],
                        Cache(), index=index)
    await enricher.enrich("Carawan Hotel, Jeddah", "facilities")
    assert index.size() == 1
    assert index.search("rooftop pool")[0]["subject"] == "Carawan Hotel, Jeddah"


@pytest.mark.asyncio
async def test_a_failed_fetch_indexes_nothing():
    index = EnrichmentIndex(SqliteVectorStore())
    enricher = Enricher([Stub(fail="rate limited")], Cache(), index=index)
    await enricher.enrich("Carawan Hotel, Jeddah", "facilities")
    assert index.size() == 0


@pytest.mark.asyncio
async def test_a_broken_index_does_not_break_the_answer():
    class Broken:
        def add(self, enrichment):
            raise RuntimeError("disk full")

    enricher = Enricher([Stub([claim("pool", "open")])], Cache(), index=Broken())
    result = await enricher.enrich("Carawan Hotel, Jeddah", "facilities")
    assert result.claims and result.claims[0].field_name == "pool"


# ---- the tool ----

@pytest.mark.asyncio
async def test_search_tool_reports_when_nothing_matches(monkeypatch):
    import web_tools
    monkeypatch.setattr(web_tools, "_index", EnrichmentIndex(SqliteVectorStore()))
    result = await web_tools.search_enrichment("anything at all")
    assert result["matches"] == [] and "nothing enriched" in result["note"]
    assert "never take price" in result["usage"]


@pytest.mark.asyncio
async def test_search_tool_rejects_an_unknown_domain(monkeypatch):
    import web_tools
    monkeypatch.setattr(web_tools, "_index", EnrichmentIndex(SqliteVectorStore()))
    with pytest.raises(ValueError):
        await web_tools.search_enrichment("pool", domain="horoscope")


# ---- keyed the way the real feed keys its snapshots ----

def test_records_carry_entity_type_and_ref():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(Enrichment(subject="Carawan Hotel, Jeddah", domain="facilities",
                         entity_type="hotel", entity_ref="Carawan Hotel",
                         claims=[claim("pool", "rooftop pool")]))
    hit = index.search("pool")[0]
    assert hit["entity_type"] == "hotel" and hit["entity_ref"] == "Carawan Hotel"


def test_search_can_be_narrowed_to_one_entity():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(Enrichment(subject="Carawan Hotel, Jeddah", domain="facilities",
                         entity_type="hotel", entity_ref="Carawan Hotel",
                         claims=[claim("pool", "rooftop pool")]))
    index.add(Enrichment(subject="Jeddah", domain="weather", entity_type="city",
                         entity_ref="Jeddah",
                         claims=[claim("forecast", "hot and dry", domain="weather")]))
    cities = index.search("pool", entity_type="city")
    assert all(h["entity_type"] == "city" for h in cities)
    assert index.search("pool", entity_ref="Carawan Hotel")[0]["entity_ref"] == "Carawan Hotel"


def test_the_same_name_under_two_entity_types_stays_separate():
    index = EnrichmentIndex(SqliteVectorStore())
    for kind in ("hotel", "city"):
        index.add(Enrichment(subject="Jeddah", domain="facilities", entity_type=kind,
                             entity_ref="Jeddah", claims=[claim("pool", f"{kind} pool")]))
    assert index.size() == 2      # entity_type is part of the key, not decoration


@pytest.mark.asyncio
async def test_the_tool_rejects_an_unknown_entity_type(monkeypatch):
    import web_tools
    monkeypatch.setattr(web_tools, "_index", EnrichmentIndex(SqliteVectorStore()))
    with pytest.raises(ValueError):
        await web_tools.search_enrichment("pool", entityType="spaceship")


# ---- questions phrased normally, not in the words we stored ----

def _weather_index():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(Enrichment(subject="Jeddah", domain="weather", entity_type="city",
                         entity_ref="Jeddah",
                         claims=[claim("forecast_2026-09-01", "29.2–34.5°C, 0 mm rain",
                                       url="https://api.open-meteo.com/x", domain="weather")]))
    index.add(Enrichment(subject="Carawan Hotel, Jeddah", domain="facilities",
                         entity_type="hotel", entity_ref="Carawan Hotel",
                         claims=[claim("pool", "rooftop pool open all year")]))
    return index


@pytest.mark.parametrize("question", [
    "how warm will it be", "temperature", "is it going to rain", "will it be hot",
])
def test_weather_answers_everyday_wording(question):
    top = _weather_index().search(question)
    assert top, f"{question!r} found nothing"
    assert top[0]["domain"] == "weather"


@pytest.mark.parametrize("question", [
    "which one has a swimming pool", "is there a gym", "somewhere with parking",
])
def test_facilities_answers_everyday_wording(question):
    top = _weather_index().search(question)
    assert top and top[0]["domain"] == "facilities"


def test_a_word_nobody_listed_still_misses():
    # the vocabulary is written down, not inferred — this is the honest limit
    from enrichment_index import expand
    assert expand("is it muggy") == "is it muggy"


def test_expansion_uses_the_domain_when_we_know_it():
    from enrichment_index import expand
    assert "temperature" in expand("anything", domain="weather")
    assert "temperature" not in expand("anything")


# ---- the floor: near-zero overlap must not come back looking like an answer ----

def test_an_unlisted_word_returns_nothing_rather_than_noise():
    """`expand` adds no vocabulary for "muggy", so the only overlap left is
    stopwords. Before the floor that scored ~0.13 and came back as a sourced
    claim about the weather, which reads exactly like a hallucination."""
    index = _weather_index()
    assert index.search("is it muggy") == []
    assert index.search("is it muggy", min_score=0.0), "still reachable if asked for"


def test_an_off_topic_question_returns_nothing():
    assert _weather_index().search("what is the price per night") == []


def test_real_matches_sit_far_above_the_floor():
    """The two bands the floor separates, asserted rather than assumed."""
    from enrichment_index import MIN_SCORE
    real = _weather_index().search("how warm will it be")
    assert real[0]["match"] > MIN_SCORE * 2, real[0]["match"]
    noise = _weather_index().search("is it muggy", min_score=0.0)
    assert noise[0]["match"] < MIN_SCORE, noise[0]["match"]


# ---------------------------------------------------------------------------
# expand() puts a whole domain vocabulary on both sides, which is what lets
# "how warm will it be" reach "29.2-34.5°C" — and also what made "the weather in
# Aswan" score 0.757 against Jeddah. The floor cannot separate one city from
# another; the names in the question can.
# ---------------------------------------------------------------------------
from enrichment_index import mentioned_entities


def _jeddah_weather():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(Enrichment(subject="Jeddah", domain="weather", entity_type="city",
                         entity_ref="Jeddah",
                         claims=[claim("forecast_2026-09-11", "29.1–35°C, 0 mm rain",
                                       url="https://api.open-meteo.com/x", domain="weather")]))
    return index


def test_a_question_about_an_unknown_city_returns_nothing():
    assert _jeddah_weather().search("What's the weather in Aswan?") == []


def test_a_question_naming_the_known_city_still_answers():
    assert _jeddah_weather().search("What is the weather in Jeddah?")


def test_a_question_naming_no_city_is_unchanged():
    """The demo's own phrasing — this is the retrieval the panel shows."""
    assert _jeddah_weather().search("how warm will it be")
    assert _jeddah_weather().search("will it rain")


def test_months_and_openers_are_not_read_as_places():
    assert mentioned_entities("What about September?") == []
    assert mentioned_entities("How warm will it be on Monday?") == []
    assert mentioned_entities("What's the weather in Aswan?") == ["Aswan"]


@pytest.mark.asyncio
async def test_the_note_names_the_subject_that_is_missing(monkeypatch):
    import web_tools
    monkeypatch.setattr(web_tools, "_index", _jeddah_weather())
    out = await web_tools.search_enrichment("What's the weather in Aswan?")
    assert out["matches"] == []
    assert "Aswan" in out["note"]


def _weather(subject, field, observed, entity_ref=None):
    e = Enrichment(subject=subject, domain="weather", entity_type="city",
                   entity_ref=entity_ref or subject,
                   claims=[claim(field, "29.1–35°C, 0 mm rain",
                                 url="https://api.open-meteo.com/x", domain="weather")])
    e.claims[0].observed_at = observed
    return e


def test_a_fresher_observation_wins_a_tie():
    """Two claims that score the same came back in SQL row order, so an older
    fetch could sit above a newer one. Across subjects, where both legitimately
    coexist, the fresher observation now leads."""
    now = datetime.now(timezone.utc)
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-02", now - timedelta(hours=2)))
    index.add(_weather("Riyadh", "forecast_2099-09-02", now))
    refs = [m["entity_ref"] for m in index.search("how warm will it be", limit=2)]
    assert refs[0] == "Riyadh", refs


def test_a_claim_past_its_domain_window_is_not_served():
    """The index used to serve whatever it had ever been told. FRESH_FOR_SECONDS
    governed only the in-process re-fetch cache, so a weather claim from
    yesterday answered today's question as readily as one from this hour."""
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-02",
                       datetime.now(timezone.utc) - timedelta(days=1)))
    assert index.search("how warm will it be") == []
    assert index.search("how warm will it be", include_stale=True), "still on record"


def test_a_fresh_claim_is_served_and_carries_its_expiry():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-02", datetime.now(timezone.utc)))
    match = index.search("how warm will it be")[0]
    assert match["is_stale"] is False
    assert match["valid_until"] > match["observed_at"]


def test_a_new_fetch_retires_the_window_before_it():
    """The reported fault: asking about one stay came back with the forecast
    fetched for another. Keyed by field, a September window and an October
    window for one city both persisted and competed."""
    now = datetime.now(timezone.utc)
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-11", now - timedelta(minutes=30)))
    index.add(_weather("Jeddah", "forecast_2099-09-02", now))
    fields = [m["field"] for m in index.search("how warm will it be", limit=5)]
    assert fields == ["forecast_2099-09-02"], fields
    assert index.size() == 1, "the earlier window is gone, not merely outranked"


def test_retiring_one_domain_leaves_the_others_alone():
    now = datetime.now(timezone.utc)
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(Enrichment(subject="Jeddah", domain="advisory", entity_type="city",
                         entity_ref="Jeddah",
                         claims=[claim("guidance", "no restrictions",
                                       url="https://travel.state.gov/x", domain="advisory")]))
    index.add(_weather("Jeddah", "forecast_2099-09-11", now - timedelta(minutes=30)))
    index.add(_weather("Jeddah", "forecast_2099-09-02", now))
    assert index.size() == 2, "the advisory claim survives a weather refetch"


def test_a_refetch_for_another_city_retires_nothing():
    now = datetime.now(timezone.utc)
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-02", now - timedelta(minutes=30)))
    index.add(_weather("Riyadh", "forecast_2099-09-02", now))
    assert index.size() == 2


def test_a_forecast_for_a_past_date_is_not_offered():
    """It stays in the index as a record. It cannot answer a question about a
    stay, so it does not take one of the caller's slots."""
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(Enrichment(subject="Jeddah", domain="weather", entity_type="city",
                         entity_ref="Jeddah",
                         claims=[claim("forecast_2020-01-01", "10–15°C, 0 mm rain",
                                       url="https://api.open-meteo.com/x", domain="weather"),
                                 claim("forecast_2099-09-02", "29–35°C, 0 mm rain",
                                       url="https://api.open-meteo.com/x", domain="weather")]))
    fields = [m["field"] for m in index.search("how warm will it be", limit=5)]
    assert "forecast_2020-01-01" not in fields
    assert "forecast_2099-09-02" in fields


@pytest.mark.asyncio
async def test_expired_and_never_fetched_read_differently(monkeypatch):
    """Both used to say "nothing enriched so far answers this". One means fetch
    it; the other means fetch it *again* — and an agent that cannot tell them
    apart will answer from data past its window."""
    import web_tools
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-02",
                       datetime.now(timezone.utc) - timedelta(days=1)))
    monkeypatch.setattr(web_tools, "_index", index)

    expired = await web_tools.search_enrichment("how warm will it be")
    assert expired["matches"] == []
    assert expired["stale_held"] == 1
    assert "freshness window" in expired["note"]
    assert "Jeddah" in expired["note"]

    unknown = await web_tools.search_enrichment("what is the pool like")
    assert unknown["matches"] == []
    assert unknown["stale_held"] == 0
    assert unknown["note"] == "nothing enriched so far answers this"


@pytest.mark.asyncio
async def test_stale_claims_are_reachable_when_explicitly_asked_for(monkeypatch):
    """The panel should be able to show what expired; the agent path should not
    get it by default."""
    import web_tools
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Jeddah", "forecast_2099-09-02",
                       datetime.now(timezone.utc) - timedelta(days=1)))
    monkeypatch.setattr(web_tools, "_index", index)
    shown = await web_tools.search_enrichment("how warm will it be", includeStale=True)
    assert len(shown["matches"]) == 1
    assert shown["matches"][0]["is_stale"] is True


def test_a_capitalised_run_is_tried_word_by_word():
    """"For Makkah, will it be hot" gave the single candidate "For Makkah",
    which matched no stored entity — so a question naming a city the index held
    retrieved nothing and the agent answered from the transcript instead."""
    candidates = mentioned_entities("For Makkah, will it be hot or rainy from 4 to 8 Sep 2026?")
    assert "Makkah" in candidates
    assert candidates.index("For Makkah") < candidates.index("Makkah"), \
        "phrases first, so a real multi-word name still wins over its parts"


def test_a_question_naming_a_known_city_mid_phrase_still_retrieves():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Makkah", "forecast_2099-09-04", datetime.now(timezone.utc)))
    assert index.search("For Makkah, will it be hot or rainy?"), "was returning nothing"
    assert index.search("What's the weather in Makkah?")


def test_an_unknown_city_inside_a_phrase_still_returns_nothing():
    index = EnrichmentIndex(SqliteVectorStore())
    index.add(_weather("Makkah", "forecast_2099-09-04", datetime.now(timezone.utc)))
    assert index.search("For Aswan, will it be hot or rainy?") == []


@pytest.mark.asyncio
async def test_truncation_is_reported_rather_than_left_to_be_inferred(monkeypatch):
    """Asked about five days, the old default of 5 returned four forecast days
    plus the place claim — so one day never reached the agent, which filled the
    gap from the day beside it."""
    import web_tools
    now = datetime.now(timezone.utc)
    index = EnrichmentIndex(SqliteVectorStore())
    for day in range(4, 9):
        index.add(_weather("Makkah", f"forecast_2099-09-0{day}", now))
    monkeypatch.setattr(web_tools, "_index", index)

    tight = await web_tools.search_enrichment("weather in Makkah", limit=3)
    assert tight["truncated"] is True
    assert tight["returned"] == 3 and tight["available"] == 5
    assert "3 of 5" in tight["note"] and "higher limit" in tight["note"]

    whole = await web_tools.search_enrichment("weather in Makkah")
    assert whole["truncated"] is False
    assert whole["returned"] == whole["available"] == 5
    assert whole["note"] is None


def test_the_prompt_requires_retrieval_first_and_row_fidelity():
    from agents.hotel_search_agent import HotelSearchAgent
    from runtime import AgentContext
    prompt = HotelSearchAgent().build_prompt(AgentContext(org_id="org-1"))
    assert "before every enrichment fetch, without exception" in prompt
    assert "never carry a neighbour's across" in prompt
    assert "name that day as missing" in prompt
