from __future__ import annotations

"""Per-session agent state: memory, the tool calls made, and a verify result."""

from dataclasses import dataclass, field
from typing import Any


class SessionMemory:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def remember(self, key: str, value: Any) -> None:
        self._data[key] = value

    def recall(self, key: str) -> Any:
        return self._data.get(key)

    def recall_all(self) -> dict[str, Any]:
        return dict(self._data)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class AgentContext:
    org_id: str
    currency: str = "USD"
    nationality: str = "AE"
    username: str | None = None
    memory: Any | None = None          # GraphitiMemory — durable, cross-session
    memory_context: str | None = None  # facts pulled at session start
    brief: str | None = None           # set when another agent delegated this work
    parent: "AgentContext | None" = field(default=None, repr=False)
    session: SessionMemory = field(default_factory=SessionMemory)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        self.session.remember(key, value)

    def recall(self, key: str) -> Any:
        return self.session.recall(key)

    def recall_all(self) -> dict[str, Any]:
        return self.session.recall_all()

    def for_child(self, brief: str) -> "AgentContext":
        """A context for an agent working on our behalf.

        It gets who the customer is and the durable store, because those belong
        to the person rather than to the conversation. It does not get our
        session keys, our tool results or our retrieved facts — a child that can
        read the parent's scratchpad ends up answering the parent's question
        instead of its own, and the transcript grows without bound.
        """
        return AgentContext(
            org_id=self.org_id, currency=self.currency, nationality=self.nationality,
            username=self.username, memory=self.memory, brief=brief, parent=self)


class VerificationResult:
    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.issues: list[str] = []

    def add_issue(self, issue: str) -> None:
        self.passed = False
        self.issues.append(issue)


"""OpenRouter chat-completions client (OpenAI-compatible, with function calling).

The agent loop only needs `complete(messages, tools) -> LLMResponse`. Tool calls
come back normalised so the loop never touches provider-specific JSON.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from config import Settings, get_settings


class LLMError(RuntimeError):
    pass


# Retried with backoff rather than raised: rate limits and capacity. Google
# documents exactly this for the Gemini endpoint, and 503 "experiencing high
# demand" clears on a second attempt more often than not.
_TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})
_BACKOFF = (1.0, 4.0)


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # The provider's own dict, kept verbatim. Gemini attaches a
    # thought_signature under extra_content and rejects the next turn without
    # it — "Function call is missing a thought_signature in functionCall parts"
    # — so a rebuilt-from-parts tool call cannot continue the conversation.
    raw: dict[str, Any] | None = None
    # Set when the model emitted arguments that are not valid JSON. The call is
    # kept rather than dropped so the loop can hand the parse error back as this
    # call's result; the model then corrects itself on the next iteration.
    invalid_arguments: str | None = None


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class OpenRouterLLM:
    def __init__(self, *, api_key: str | None, model: str,
                 base_url: str = "https://openrouter.ai/api/v1",
                 max_tokens: int | None = None,
                 timeout: float = 60.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    @property
    def host(self) -> str:
        """What to name in an error. Four providers share this client now, so
        "OpenRouter HTTP 503" was wrong for three of them."""
        from urllib.parse import urlparse
        return (urlparse(self.base_url).netloc or self.base_url).removeprefix("www.")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        base: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            base["tools"] = tools
            base["tool_choice"] = "auto"

        # One retry: OpenRouter returns 402 with "can only afford N tokens" when the
        # request's max_tokens exceeds the caller's credit balance. Parse N and retry
        # with that budget so the session self-heals as credits drain.
        # Each recovery fires once, tracked by its own flag rather than by the
        # attempt number. Gating the budget retry on `attempt == 0` meant a
        # spelling flip on the first attempt used up its only chance, so a 402
        # arriving second was raised untouched — which is how a key that could
        # afford 527 reported "you requested up to 4000" and no retry at all.
        tried_other_spelling = False
        tried_budget = False
        transient_retries = 0
        original_model = self.model
        first_failure: tuple[int, str] | None = None
        # Enough attempts for two transient backoffs plus a budget trim plus a
        # spelling flip; every path sets a flag, so none of them can loop.
        for attempt in range(5):
            body = dict(base)
            body["model"] = self.model
            if self.max_tokens:
                body["max_tokens"] = self.max_tokens
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                # ReadTimeout stringifies to "", so the old message ended at the
                # colon and said nothing at all. Name the class and the host.
                raise LLMError(
                    f"{self.host} request failed: {type(exc).__name__}"
                    f"{f': {exc}' if str(exc) else ''}") from exc

            if resp.status_code == 402 and not tried_budget:
                afford = re.search(r"can only afford (\d+)", resp.text)
                if afford:
                    tried_budget = True
                    # Honour the number OpenRouter gave. The old floor of 256
                    # guaranteed a second 402 whenever the affordable budget was
                    # below it — measured on a key that could afford 46: the
                    # retry asked for 256, was refused again, and the page said
                    # "no credit left" about a key that answers fine at 38.
                    self.max_tokens = max(16, int(afford.group(1)) - 8)
                    continue

            # A name listed with :free on one account and without it on another
            # comes back as "no such model". Separately, a :free name whose
            # shared pool is saturated answers 429 while the plain spelling
            # answers 200 — measured on z-ai/glm-5.2 — and the reverse holds for
            # a paid name on an account with no credit. Try the other spelling
            # once in either case; the flag stops it looping.
            # "No endpoints found for <id>." is the 404 for a name that exists
            # only in the other spelling — inclusionai/ling-3.0-flash-fin is
            # published as :free and nothing else — and it never says "model".
            # Only on OpenRouter: :free is its naming convention, and appending
            # it elsewhere turns a readable error into "the model
            # `groq/compound:free` does not exist", which sends you hunting for
            # a model that was never the problem.
            # Transient: capacity and rate limits. Google documents backoff
            # for 429/408/5xx, and a single 503 "experiencing high demand" was
            # being surfaced as a dead end after exactly one attempt.
            if resp.status_code in _TRANSIENT and transient_retries < 2:
                transient_retries += 1
                await asyncio.sleep(_BACKOFF[transient_retries - 1])
                continue

            # Only a naming 404/400 flips the spelling. Flipping on 429 was a
            # gamble that could land on the paid twin of a free model, and a
            # rate limit is not a naming problem — it is retried above instead.
            if (not tried_other_spelling
                    and "openrouter.ai" in self.base_url
                    and resp.status_code in (400, 404)
                    and re.search(r"model|no endpoints", resp.text, re.I)):
                tried_other_spelling = True
                first_failure = (resp.status_code, resp.text)
                self.model = free_variant(self.model)
                continue

            if resp.status_code >= 400:
                # A flip off a rate-limited :free name can land on a spelling the
                # account cannot reach, turning "429, try again" into "No
                # endpoints found for inclusionai/ling-3.0-flash-fin" — a name
                # nobody configured. Report the failure that actually happened.
                if first_failure and re.search(
                        r"no endpoints|does not exist|not a valid model", resp.text, re.I):
                    status, text = first_failure
                    self.model = original_model
                    raise LLMError(f"{self.host} HTTP {status} for {original_model}: {text[:400]}")
                raise LLMError(f"{self.host} HTTP {resp.status_code}: {resp.text[:500]}")
            break

        data = resp.json()
        if data.get("error"):
            raise LLMError(str(data["error"]))

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"OpenRouter returned no choices: {str(data)[:300]}")
        message = choices[0].get("message") or {}

        calls: list[LLMToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            invalid = None
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                # Raising here ended the whole turn on one malformed tool call —
                # seen live as "tool call arguments were not valid JSON:
                # Expecting ',' delimiter", with the conversation dead and the
                # tools already run. A model that mis-serialises one call can
                # fix it if it is told; it cannot if the turn is gone.
                args, invalid = {}, getattr(exc, "msg", str(exc))
            calls.append(LLMToolCall(id=tc.get("id") or fn.get("name", ""), name=fn.get("name", ""),
                                     arguments=args, raw=tc, invalid_arguments=invalid))

        return LLMResponse(content=message.get("content"), tool_calls=calls)


def free_variant(model: str) -> str:
    """Some models are listed with a :free suffix and some without, and which one
    an account can reach differs. Give the other spelling of the same name."""
    return model[:-5] if model.endswith(":free") else model + ":free"


def build_llm(settings: Settings | None = None, model: str | None = None) -> OpenRouterLLM:
    s = settings or get_settings()
    if s.llm_provider != "openrouter":
        raise LLMError(f"unsupported LLM_PROVIDER {s.llm_provider!r}; only 'openrouter' is implemented")
    chosen, api_key = s.credentials_for(model)
    # 60s is not enough for a reasoning model behind a tool loop: Gemini returns
    # a thought_signature blob per call and timed out mid-turn, surfacing as
    # "OpenRouter request failed:" with nothing after the colon.
    return OpenRouterLLM(api_key=api_key, model=chosen, timeout=180.0,
                         base_url=s.base_url_for(chosen), max_tokens=s.openrouter_max_tokens)


"""The hotel tools an agent may call. Read and draft only: search, reprice, and
read bookings. Booking and cancel are intentionally not here — they can't be
called because they don't exist in this map.

organizationId / currency / nationality are injected from the agent context, so
the model neither sees nor sets them (it never learns the org id).
"""

import inspect
from typing import Any

from pydantic import BaseModel

import hotel_tools as hotels
import memory_tools
import web_tools
import ops_tools as ops

_IMPL = {
    "resolve_destination": hotels.resolve_destination,
    "search_hotel_availability": hotels.search_hotel_availability,
    "refresh_hotel_price": hotels.refresh_hotel_price,
    "list_hotel_bookings": hotels.list_hotel_bookings,
    "get_hotel_booking": hotels.get_hotel_booking,
    "get_hotel_search_results": hotels.get_hotel_search_results,
    "poll_hotel_booking": hotels.poll_hotel_booking,
    "get_hotel_static_data": hotels.get_hotel_static_data,
    "get_hotel_availability_options": hotels.get_hotel_availability_options,
    "get_hotel_options": hotels.get_hotel_options,
    "remember_preference": memory_tools.remember_preference,
    "recall_preferences": memory_tools.recall_preferences,
    "record_ops_pattern": memory_tools.record_ops_pattern,
    "enrich_hotel_info": web_tools.enrich_hotel_info,
    "enrich_destination": web_tools.enrich_destination,
    "search_enrichment": web_tools.search_enrichment,
}

TOOL_SPECS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "resolve_destination",
        "description": "Turn a city or hotel name into destination code(s). Prefer CITY results; use to disambiguate.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "City or hotel name"},
            "limit": {"type": "integer", "default": 10},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_hotel_availability",
        "description": "Search hotels for a city and date range. Returns sorted, filtered, priced hotels plus destination alternatives and paging info.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "checkIn": {"type": "string", "description": "YYYY-MM-DD"},
            "checkOut": {"type": "string", "description": "YYYY-MM-DD"},
            "adults": {"type": "integer", "default": 2},
            "childrenAges": {"type": "array", "items": {"type": "integer"}},
            "roomCount": {"type": "integer", "default": 1},
            "destinationCode": {"type": "string", "description": "Force a specific destination code (from resolve_destination) to skip name matching."},
            "sortField": {"type": "string", "enum": ["PRICE", "RATING", "RECOMMENDED"], "default": "PRICE",
                          "description": "RATING is the star rating. There is no STARS value."},
            "sortOrder": {"type": "string", "enum": ["asc", "desc"], "default": "asc"},
            "minPrice": {"type": "number", "description": "Lowest total price to include."},
            "maxPrice": {"type": "number", "description": "Highest total price to include."},
            "minStars": {"type": "integer", "description": "Lowest star rating to include."},
            "maxStars": {"type": "integer", "description": "Highest star rating to include."},
            "amenities": {"type": "array", "items": {"type": "string"},
                          "description": "Only hotels having all of these, e.g. [\"wifi\", \"pool\"]."},
            "pageNumber": {"type": "integer", "default": 0},
            "limit": {"type": "integer", "default": 5, "description": "How many hotels to return."},
        }, "required": ["city", "checkIn", "checkOut"]}}},
    {"type": "function", "function": {
        "name": "get_hotel_search_results",
        "description": "Read a page of an existing search by its uuid — to let prices finish loading, to re-sort or re-filter, or to page. Does not start a new search.",
        "parameters": {"type": "object", "properties": {
            "uuid": {"type": "string"},
            "sortField": {"type": "string", "enum": ["PRICE", "RATING", "RECOMMENDED"], "default": "PRICE",
                          "description": "RATING is the star rating. There is no STARS value."},
            "sortOrder": {"type": "string", "enum": ["asc", "desc"], "default": "asc"},
            "minPrice": {"type": "number", "description": "Lowest total price to include."},
            "maxPrice": {"type": "number", "description": "Highest total price to include."},
            "minStars": {"type": "integer", "description": "Lowest star rating to include."},
            "maxStars": {"type": "integer", "description": "Highest star rating to include."},
            "amenities": {"type": "array", "items": {"type": "string"},
                          "description": "Only hotels having all of these, e.g. [\"wifi\", \"pool\"]."},
            "pageNumber": {"type": "integer", "default": 0},
            "pageSize": {"type": "integer", "default": 20},
        }, "required": ["uuid"]}}},
    {"type": "function", "function": {
        "name": "get_hotel_static_data",
        "description": "Content for one hotel: name, address, star rating, media, phones. Use after the user picks a hotel from the results.",
        "parameters": {"type": "object", "properties": {
            "hotelCode": {"type": "string"},
            "extras": {"type": "array", "items": {"type": "string", "enum": ["descriptions", "facilities"]},
                       "description": "facilities are the hotel amenities."},
        }, "required": ["hotelCode"]}}},
    {"type": "function", "function": {
        "name": "get_hotel_availability_options",
        "description": "Room options for one hotel inside a running search (needs the search uuid): board/meal plan, price, cancellation policy. Filter by refundableOnly, mealPlan, price.",
        "parameters": {"type": "object", "properties": {
            "uuid": {"type": "string", "description": "The search uuid."},
            "hotelCode": {"type": "string"},
            "refundableOnly": {"type": "boolean", "default": False},
            "mealPlan": {"type": "array", "items": {"type": "string"},
                         "description": "Match board text/code, e.g. [\"breakfast\"]."},
            "minPrice": {"type": "number"},
            "maxPrice": {"type": "number"},
        }, "required": ["uuid", "hotelCode"]}}},
    {"type": "function", "function": {
        "name": "get_hotel_options",
        "description": "Priced room options for one hotel, outside any search session. Each record carries the option id that refresh_hotel_price takes as optionRefId. Use when the search uuid has expired.",
        "parameters": {"type": "object", "properties": {
            "hotelCode": {"type": "string"},
            "checkIn": {"type": "string", "description": "YYYY-MM-DD"},
            "checkOut": {"type": "string", "description": "YYYY-MM-DD"},
            "adults": {"type": "integer", "default": 2},
            "childrenAges": {"type": "array", "items": {"type": "integer"}},
            "roomCount": {"type": "integer", "default": 1},
            "refundableOnly": {"type": "boolean", "default": False},
            "minPrice": {"type": "number"},
            "maxPrice": {"type": "number"},
        }, "required": ["hotelCode", "checkIn", "checkOut"]}}},
    {"type": "function", "function": {
        "name": "refresh_hotel_price",
        "description": "Confirm the live rate for a selected room before quoting it. Does not book.",
        "parameters": {"type": "object", "properties": {
            "optionRefId": {"type": "string"},
            "applyMarkup": {"type": "boolean", "default": False},
        }, "required": ["optionRefId"]}}},
    {"type": "function", "function": {
        "name": "list_hotel_bookings",
        "description": "List existing hotel bookings for the organization, newest first.",
        "parameters": {"type": "object", "properties": {
            "bookingStatus": {"type": "string"},
            "customerId": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        }}}},
    {"type": "function", "function": {
        "name": "get_hotel_booking",
        "description": "Get one hotel booking by its numeric Id.",
        "parameters": {"type": "object", "properties": {
            "Id": {"type": "integer"},
        }, "required": ["Id"]}}},
    {"type": "function", "function": {
        "name": "poll_hotel_booking",
        "description": "Check a booking's status by its HotelBookingId string.",
        "parameters": {"type": "object", "properties": {
            "hotelBookingId": {"type": "string"},
        }, "required": ["hotelBookingId"]}}},
]

AVAILABLE_TOOL_NAMES = frozenset(_IMPL)

# --- ops / booking-queue tools (used by the ops triage agent) ---
_IMPL.update({
    "get_queue_summary": ops.get_queue_summary,
    "get_failed_messages": ops.get_failed_messages,
    "get_message_detail": ops.get_message_detail,
    "get_transaction": ops.get_transaction,
    "list_transactions": ops.list_transactions,
    "run_named_query": ops.run_named_query,
})
TOOL_SPECS.extend([
    {"type": "function", "function": {
        "name": "enrich_hotel_info",
        "description": "What the web says about one hotel: reputation, location, facilities, and risks such as renovation or closure. Every claim carries its sources and whether they agree. Never a source of price, availability or cancellation terms.",
        "parameters": {"type": "object", "properties": {
            "hotelName": {"type": "string"},
            "city": {"type": "string"},
            "domains": {"type": "array", "items": {"type": "string",
                        "enum": ["reputation", "location", "facilities", "risk"]}},
            "officialSite": {"type": "string", "description": "Hotel's own domain, so its pages outrank aggregators."},
            "pageUrl": {"type": "string", "description": "One page to read in a browser, for a site that renders nothing without JavaScript. Only used when WEB_PLAYWRIGHT_ENABLED is on."},
        }, "required": ["hotelName"]}}},
    {"type": "function", "function": {
        "name": "enrich_destination",
        "description": "Conditions at the destination for the stay: weather for the dates, current travel advice, and recent news. Weather comes from a forecast API, not from prose.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "checkIn": {"type": "string", "description": "YYYY-MM-DD"},
            "checkOut": {"type": "string", "description": "YYYY-MM-DD"},
            "domains": {"type": "array", "items": {"type": "string",
                        "enum": ["weather", "advisory", "news"]}},
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_enrichment",
        "description": "Ask in plain words across everything already fetched about hotels and destinations, without knowing the subject or domain first. Use it before fetching again — it costs nothing and may already hold the answer.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "entityType": {"type": "string", "enum": ["hotel", "city"]},
            "entityRef": {"type": "string", "description": "The hotel or city name."},
            "subject": {"type": "string", "description": "Loose match on the subject line."},
            "domain": {"type": "string", "enum": ["reputation", "location", "facilities",
                       "risk", "weather", "advisory", "news"]},
            "limit": {"type": "integer", "default": 5},
        }, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "remember_preference",
        "description": "Store something the user stated about how they like to travel or book, so it is still known in future sessions. Only call this when the user states a preference — never for search results or supplier data.",
        "parameters": {"type": "object", "properties": {
            "statement": {"type": "string", "description": "The preference in the user's own words."},
            "key": {"type": "string", "description": "What the preference is about, e.g. hotel_stars, board, city. A new value for the same key supersedes the old one."},
        }, "required": ["statement"]}}},
    {"type": "function", "function": {
        "name": "recall_preferences",
        "description": "Look up what is already known about this user and organization for a topic.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "record_ops_pattern",
        "description": "Record a dead-letter error signature, and report whether it was already known from an earlier session.",
        "parameters": {"type": "object", "properties": {
            "signature": {"type": "string", "description": "The error signature, e.g. operation:classification."},
        }, "required": ["signature"]}}},
    {"type": "function", "function": {
        "name": "get_queue_summary",
        "description": "Count booking-queue messages by status (pending, processing, complete, failed).",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 1000}}}}},
    {"type": "function", "function": {
        "name": "get_failed_messages",
        "description": "List dead-letter (failed) queue messages, newest first.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}}},
    {"type": "function", "function": {
        "name": "get_message_detail",
        "description": "Full detail for one queue message including the error trace.",
        "parameters": {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]}}},
    {"type": "function", "function": {
        "name": "get_transaction",
        "description": "Load the transaction linked to a message, by its TransactionGuid.",
        "parameters": {"type": "object", "properties": {
            "guid": {"type": "string"},
            "fields": {"type": "array", "items": {"type": "string"}}}, "required": ["guid"]}}},
    {"type": "function", "function": {
        "name": "list_transactions",
        "description": "Transactions for the organization in a time window (org scope enforced).",
        "parameters": {"type": "object", "properties": {
            "where": {"type": "object"}, "limit": {"type": "integer", "default": 50}}}}},
    {"type": "function", "function": {
        "name": "run_named_query",
        "description": "Run a named query. 'triage_context' returns a failed message plus its linked transaction.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "variables": {"type": "object"}}, "required": ["name"]}}},
])
AVAILABLE_TOOL_NAMES = frozenset(_IMPL)


def specs_for(names: frozenset[str] | set[str]) -> list[dict[str, Any]]:
    return [s for s in TOOL_SPECS if s["function"]["name"] in names]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


async def dispatch(name: str, args: dict[str, Any] | None, ctx: AgentContext) -> Any:
    if name not in _IMPL:
        raise KeyError(f"tool {name!r} is not available to this agent")
    fn = _IMPL[name]
    params = inspect.signature(fn).parameters
    call_args = dict(args or {})
    if "organizationId" in params:
        call_args.setdefault("organizationId", ctx.org_id)
    if "currency" in params:
        call_args.setdefault("currency", ctx.currency)
    if "nationality" in params:
        call_args.setdefault("nationality", ctx.nationality)
    if "ctx" in params:
        call_args["ctx"] = ctx
    return _jsonable(await fn(**call_args))


"""Agent runtime: build a system prompt, run the LLM tool-calling loop against
the allowed tools, then verify. Providers and tools are injected, so subclasses
only declare role, prompt, tool set, and checks.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any



@dataclass
class Handover:
    """What comes back from an agent that worked on another one's behalf. The
    answer and how it was reached — not the messages, not the tool payloads."""
    agent: str
    answer: str
    tools_used: list[str]
    passed: bool
    issues: list[str] = field(default_factory=list)

    def to_model(self) -> dict[str, Any]:
        return {"agent": self.agent, "answer": self.answer,
                "tools_used": self.tools_used, "verified": self.passed,
                "issues": self.issues}


async def delegate(agent: "AgentBase", brief: str, llm, parent: AgentContext,
                   max_iterations: int = 6) -> Handover:
    """Run another agent on its own context and bring back a summary.

    The child builds its own prompt from the brief, keeps its own memory and its
    own tool calls, and hands back one paragraph. Nothing it read reaches the
    caller's history.
    """
    child = parent.for_child(brief)
    result = await agent.run(child, brief, llm, max_iterations=max_iterations)
    return Handover(agent=agent.get_role(), answer=result.output,
                    tools_used=[c.name for c in child.tool_calls],
                    passed=result.verification.passed,
                    issues=list(result.verification.issues))


@dataclass
class AgentRunResult:
    output: str
    verification: VerificationResult
    context: AgentContext
    messages: list[dict[str, Any]]


class AgentBase(ABC):
    @abstractmethod
    def get_role(self) -> str: ...

    @abstractmethod
    def allowed_tools(self) -> frozenset[str]: ...

    @abstractmethod
    def build_prompt(self, ctx: AgentContext) -> str: ...

    def on_tool_result(self, ctx: AgentContext, call: ToolCall) -> None:
        """Persist anything worth keeping from a tool result. Default: nothing."""

    def on_run_start(self, ctx: AgentContext) -> None:
        """Called once at the start of a run. Default: nothing."""

    def on_run_end(self, ctx: AgentContext, output: str) -> None:
        """Called once after the loop, before verify. Default: nothing."""

    @abstractmethod
    async def verify(self, ctx: AgentContext) -> VerificationResult: ...

    async def run(self, ctx: AgentContext, user_message: str, llm,
                  max_iterations: int = 8,
                  history: list[dict[str, Any]] | None = None) -> AgentRunResult:
        specs = specs_for(self.allowed_tools())
        if ctx.memory is not None and ctx.memory_context is None:
            # one retrieval per session, cached on the context for its lifetime
            ctx.memory_context = await ctx.memory.get_context(
                user_message, username=ctx.username, org_id=ctx.org_id)
        if history:
            # Continue an existing conversation (multi-turn chat). The system
            # prompt is already at the front of `history`.
            messages = list(history)
            messages.append({"role": "user", "content": user_message})
        else:
            system = self.build_prompt(ctx)
            if ctx.brief is not None:
                system += ("\n\nAnother agent asked you for this. You cannot see its "
                           "conversation, so work only from the request above. Answer in "
                           "one short paragraph it can use directly — findings and what "
                           "they rest on, no preamble.")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]
        self.on_run_start(ctx)
        output = None
        for _ in range(max_iterations):
            resp = await llm.complete(messages, tools=specs)
            if not resp.tool_calls:
                # Stopping is not the same as answering. Models that reason in a
                # separate field return content="" and put everything there, so
                # taking the empty string ended the turn on "(no reply)" — with
                # the tools already called and the badge still green. Leave
                # `output` unset and let the fallback below ask for words.
                if (resp.content or "").strip():
                    output = resp.content
                    # The answer has to go into `messages` too. `messages` becomes
                    # the session history, so leaving it out meant the next turn saw
                    # two consecutive user turns and no record of what the agent had
                    # said — "which one was cheapest?" had nothing to refer back to.
                    messages.append({"role": "assistant", "content": output})
                break
            messages.append({
                "role": "assistant", "content": resp.content or "",
                # Echo the provider's own tool_call dict when we have it, so
                # fields we do not model (Gemini's thought_signature) survive
                # the round trip.
                "tool_calls": [tc.raw or {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                } for tc in resp.tool_calls],
            })
            for tc in resp.tool_calls:
                if tc.invalid_arguments is not None:
                    # Not dispatched and not recorded as a tool call — it never
                    # ran. The model is told what was wrong and tries again.
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps({"error": (
                            f"arguments for {tc.name} were not valid JSON "
                            f"({tc.invalid_arguments}). Send the call again with "
                            "well-formed JSON arguments.")}),
                    })
                    continue
                try:
                    result = await dispatch(tc.name, tc.arguments, ctx)
                except Exception as exc:  # surface the failure to the model, keep going
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                call = ToolCall(name=tc.name, args=tc.arguments, result=result)
                ctx.tool_calls.append(call)
                self.on_tool_result(ctx, call)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
        if output is None:
            # Either the tool-call cap was hit, or the model stopped without
            # writing anything. Force an answer from what was gathered so the
            # user gets a real reply, never a dead end.
            messages.append({"role": "user", "content":
                "Give your final answer now using the information you already have. "
                "Do not call any tools. If no priced hotels were found, say so and "
                "suggest different dates."})
            resp = await llm.complete(messages, tools=None)
            # Last resort, and it must not invent a cause. The old wording said
            # no priced availability was found — printed verbatim after
            # get_hotel_availability_options had returned four priced rooms,
            # because the only thing that actually failed was the write-up.
            output = resp.content or (
                "I gathered the information for that but did not manage to write it "
                "up. The tool results are still in this session — ask again and I "
                "will answer from them without searching afresh.")
            messages.append({"role": "assistant", "content": output})
        self.on_run_end(ctx, output)
        verification = await self.verify(ctx)
        return AgentRunResult(output=output, verification=verification, context=ctx, messages=messages)
