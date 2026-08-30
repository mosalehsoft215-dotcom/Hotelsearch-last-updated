"""The defects a live run exposed, and the model options the page offers."""
import httpx
import pytest

from config import Settings
from runtime import build_llm, free_variant
from web_enrich import MONEY, OpenRouterWeb, ProviderUnavailable


def _provider(content, urls, status=200):
    body = {"choices": [{"message": {
        "content": content,
        "annotations": [{"type": "url_citation", "url_citation": {"url": u}} for u in urls]}}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    return OpenRouterWeb(api_key="k", base_url="https://openrouter.ai/api/v1",
                         model="m", transport=transport)


# ---- money, in both word orders ----

def test_money_is_caught_whichever_side_the_currency_sits():
    for text in ("$420", "USD 420", "SAR 1500", "AED 700", "€99",
                 "420 USD", "380 EUR", "1500 SAR", "700 AED", "420 dollars", "420 riyals"):
        assert MONEY.search(text), f"{text} should read as money"


def test_ordinary_values_are_not_mistaken_for_money():
    for text in ("29–38°C, 0 mm rain", "8.7 out of 10", "2300 reviews",
                 "five minute walk", "closed for refurbishment"):
        assert not MONEY.search(text), f"{text} should not read as money"


@pytest.mark.asyncio
async def test_a_suffix_price_is_dropped_while_parsing():
    content = ("nightly | 420 USD per night | https://booking.com/a\n"
               "pool | open all year | https://booking.com/a")
    claims = await _provider(content, ["https://booking.com/a"]).fetch("H", "facilities", {})
    assert [c.field_name for c in claims] == ["pool"]


@pytest.mark.asyncio
async def test_the_agent_flags_a_suffix_price_that_reached_it():
    from agents.hotel_search_agent import HotelSearchAgent
    from runtime import AgentContext, ToolCall

    ctx = AgentContext(org_id="org-1")
    ctx.tool_calls = [ToolCall("enrich_hotel_info", {"hotelName": "H"},
                               {"domains": {"facilities": {"findings": {
                                   "rooms": [{"value": "420 USD per night"}]}}}})]
    outcome = await HotelSearchAgent().verify(ctx)
    assert not outcome.passed and any("price" in i for i in outcome.issues)


# ---- the citation check has to fail closed ----

@pytest.mark.asyncio
async def test_no_annotations_means_no_claims():
    content = "guest_rating | 8.7 out of 10 | https://invented.example/page"
    assert await _provider(content, []).fetch("H", "reputation", {}) == []


@pytest.mark.asyncio
async def test_a_url_the_search_never_returned_is_dropped():
    content = ("guest_rating | 8.7 | https://booking.com/a\n"
               "praised_for | quiet | https://invented.example/x")
    claims = await _provider(content, ["https://booking.com/a"]).fetch("H", "reputation", {})
    assert [c.field_name for c in claims] == ["guest_rating"]


# ---- models answer in markdown tables ----

@pytest.mark.asyncio
async def test_a_markdown_table_is_read_not_discarded():
    content = ("| Field | Value | URL |\n"
               "| :--- | :--- | :--- |\n"
               "| guest_rating | 4.0 out of 5 | [tripadvisor](https://booking.com/a) |\n")
    claims = await _provider(content, ["https://booking.com/a"]).fetch("H", "reputation", {})
    assert len(claims) == 1
    assert claims[0].field_name == "guest_rating" and claims[0].value == "4.0 out of 5"
    assert claims[0].sources[0].url == "https://booking.com/a"


@pytest.mark.asyncio
async def test_plain_lines_still_work():
    content = "guest_rating | 8.7 out of 10 | https://booking.com/a"
    claims = await _provider(content, ["https://booking.com/a"]).fetch("H", "reputation", {})
    assert claims[0].value == "8.7 out of 10"


@pytest.mark.asyncio
async def test_the_divider_row_is_not_read_as_a_claim():
    content = "| :--- | :--- | :--- |\n| pool | open | https://booking.com/a |"
    claims = await _provider(content, ["https://booking.com/a"]).fetch("H", "facilities", {})
    assert [c.field_name for c in claims] == ["pool"]


# ---- gemini is gone ----

def test_no_gemini_provider_remains():
    import web_enrich
    assert not hasattr(web_enrich, "GeminiGrounded")
    settings = Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None)
    assert [p.name for p in web_enrich.build_providers(settings)] == ["open-meteo"]


# ---- three models, three keys ----

def _three() -> Settings:
    return Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None,
                    OPENROUTER_MODEL="anthropic/claude-haiku-4.5", OPENROUTER_API_KEY="key-a",
                    OPENROUTER_MODEL_B="poolside/laguna-xs-2.1", OPENROUTER_API_KEY_B="key-b",
                    OPENROUTER_MODEL_C="google/gemma-4-31b-it", OPENROUTER_API_KEY_C="key-c")


def test_the_page_is_offered_every_configured_model_default_first():
    assert [o["model"] for o in _three().model_options()] == [
        "anthropic/claude-haiku-4.5", "poolside/laguna-xs-2.1", "google/gemma-4-31b-it"]


def test_a_model_without_a_key_is_not_offered():
    settings = Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None,
                        OPENROUTER_MODEL="anthropic/claude-haiku-4.5", OPENROUTER_API_KEY="key-a",
                        OPENROUTER_MODEL_B="poolside/laguna-xs-2.1", OPENROUTER_API_KEY_B=None)
    assert [o["model"] for o in settings.model_options()] == ["anthropic/claude-haiku-4.5"]


def test_each_model_is_paid_for_by_its_own_key():
    settings = _three()
    assert settings.credentials_for("poolside/laguna-xs-2.1") == ("poolside/laguna-xs-2.1", "key-b")
    assert settings.credentials_for("google/gemma-4-31b-it") == ("google/gemma-4-31b-it", "key-c")


def test_an_unknown_model_falls_back_to_the_default_not_someone_elses_key():
    assert _three().credentials_for("some/other-model")[1] == "key-a"
    assert _three().credentials_for(None)[0] == "anthropic/claude-haiku-4.5"


def test_build_llm_picks_up_the_requested_model():
    llm = build_llm(_three(), model="poolside/laguna-xs-2.1")
    assert llm.model == "poolside/laguna-xs-2.1"


def test_the_free_suffix_flips_both_ways():
    assert free_variant("poolside/laguna-xs-2.1") == "poolside/laguna-xs-2.1:free"
    assert free_variant("poolside/laguna-xs-2.1:free") == "poolside/laguna-xs-2.1"


@pytest.mark.asyncio
async def test_a_rejected_model_name_is_retried_with_the_other_spelling():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        model = _json.loads(request.content.decode())["model"]
        seen.append(model)
        if not model.endswith(":free"):
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    from runtime import OpenRouterLLM
    llm = OpenRouterLLM(api_key="k", model="poolside/laguna-xs-2.1",
                        transport=httpx.MockTransport(handler))
    reply = await llm.complete([{"role": "user", "content": "hi"}])
    assert reply.content == "ok"
    assert seen == ["poolside/laguna-xs-2.1", "poolside/laguna-xs-2.1:free"]
    await llm.aclose()


def test_a_slot_can_live_on_a_different_host():
    """A gsk_ key is a Groq key; openrouter.ai answers it with "Missing
    Authentication header" because it is not an OpenRouter key at all."""
    settings = Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None,
                        OPENROUTER_MODEL="anthropic/claude-haiku-4.5", OPENROUTER_API_KEY="key-a",
                        OPENROUTER_MODEL_F="groq/compound", OPENROUTER_API_KEY_F="gsk_x",
                        OPENROUTER_BASE_URL_F="https://api.groq.com/openai/v1")
    assert settings.base_url_for("groq/compound") == "https://api.groq.com/openai/v1"
    assert settings.base_url_for("anthropic/claude-haiku-4.5") == settings.openrouter_base_url
    assert settings.base_url_for("some/unknown") == settings.openrouter_base_url
    assert build_llm(settings, model="groq/compound").base_url == "https://api.groq.com/openai/v1"
    # The host travels with the slot; the key still pays only for its own model.
    assert settings.credentials_for("groq/compound") == ("groq/compound", "gsk_x")
