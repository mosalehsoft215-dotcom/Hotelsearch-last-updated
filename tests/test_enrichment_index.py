"""Free-text search across what enrichment already fetched."""
from datetime import datetime, timezone

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


def test_a_fresher_observation_wins_a_tie():
    """Two forecast windows for one city scored 0.739 each and came back
    interleaved in row order, so yesterday's fetch for the wrong week sat above
    today's for the right one."""
    index = EnrichmentIndex(SqliteVectorStore())
    stale = Enrichment(subject="Jeddah", domain="weather", entity_type="city",
                       entity_ref="Jeddah",
                       claims=[claim("forecast_2099-09-11", "29.1–35°C, 0 mm rain",
                                     url="https://api.open-meteo.com/x", domain="weather")])
    stale.claims[0].observed_at = datetime(2026, 8, 30, tzinfo=timezone.utc)
    fresh = Enrichment(subject="Jeddah", domain="weather", entity_type="city",
                       entity_ref="Jeddah",
                       claims=[claim("forecast_2099-09-02", "29.1–35°C, 0 mm rain",
                                     url="https://api.open-meteo.com/x", domain="weather")])
    fresh.claims[0].observed_at = datetime(2026, 8, 31, tzinfo=timezone.utc)
    index.add(stale)
    index.add(fresh)
    fields = [m["field"] for m in index.search("how warm will it be", limit=2)]
    assert fields[0] == "forecast_2099-09-02", fields


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
