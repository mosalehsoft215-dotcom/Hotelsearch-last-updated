# hotels_mcp — hotel search & ops triage agents

Two read/draft-only agents for the TripOn/Rihla platform, on a shared runtime.
Neither writes, books, cancels, or replays anything — the tools that would are
absent from each agent's set, so it is enforced by construction rather than by
prompting.

- **Hotel search** — finds hotels, shows room options, and locks a live rate for
  a quotation.
- **Ops triage** — triages failed booking-queue (dead-letter) messages into a
  report.

## Layout

    config.py                settings, read from .env
    hasura.py                GraphQL client, response envelope, models, auth token
    hotel_tools.py           hotel tools: search, filters, hotel detail, reprice, reads
    web_enrich.py            web context for a hotel: backends, citations, injection guard
    web_tools.py             enrich_hotel_info, enrich_destination, enrich_company_facts,
                             enrich_agency_facts, search_enrichment
    enrichment_index.py      SQLite vector index behind search_enrichment
    ops_tools.py             ops tools: queue summary, failed messages, transactions
    memory.py                durable memory (Layer 2): local graph or Graphiti/FalkorDB
    memory_tools.py          remember_preference, recall_preferences, record_ops_pattern
    runtime.py               agent context/memory, OpenRouter client, tool loop, registry
    agents/
      hotel_search_agent.py
      ops_triage_agent.py
    api.py                   web server: /chat, /memory, /enrichment, /delegate, /health
    chat_ui.html             two-agent console + memory, enrichment and delegation panels
    login.py                 loginRihla -> JWT (+ decode user id)
    healthcheck.py           connection, admin-secret access, JWT login
    check_room_options.py    static data + priced options for one hotel, no agent involved
    demo_memory.py           four durable-memory scenes, no infrastructure needed
    scripts/introspect.py    read the live GraphQL schema
    tests/                   496 hermetic tests + gated live checks

## Run

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    copy .env.example .env          # then fill it in
    uvicorn api:app --reload        # http://127.0.0.1:8000
    pytest -q

macOS / Linux:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env            # then fill it in
    uvicorn api:app --reload
    pytest -q

Check the backend before blaming the agent:

    python healthcheck.py                                   endpoint, admin secret, JWT login
    python check_room_options.py 1442211 2026-09-10 2026-09-13

Programmatic:

    from agents.hotel_search_agent import answer
    from agents.ops_triage_agent import answer_triage
    await answer("Find a hotel for a city and dates", org_id="<org>")
    await answer_triage("Triage the failed booking queue", org_id="<org>")

## Configuration

Everything comes from `.env`; `config.py` is the list. `.env.example` is a
complete template — copy it, do not commit the filled copy.

The values that stop the service from working if they are wrong:

    YARVEL_URL / YARVEL_SECRET / YARVEL_ORG_ID    the Hasura endpoint and tenant
    OPENROUTER_API_KEY / OPENROUTER_MODEL         the default model and the key that pays for it
    TRIPON_SENDER_IP                              only for the JWT path; loginRihla rejects requests without it

`OPENROUTER_MODEL` is the console's default and the key must be on an account
that can pay for that model — OpenRouter answers a key with no credits with
`HTTP 402 Insufficient credits`, and the console shows that text as-is. Put a
model the key can actually reach first, or switch model in the page.

## Auth

Dev uses the admin secret (`HOTELS_AUTH_MODE=admin_secret`). Production forwards
the caller's JWT (`forward_jwt`); get one with `python login.py`. `loginRihla`
needs the admin secret, a `sender-ip` header, and `origin=agency`.

## Roles

Each agent declares the capability modules it is granted, in `GRANTED_MODULES`:

    hotel_search_agent : bookings, queries, quotations, flights, hotels
    ops_triage_agent   : bookings, queries, ops

These name the modules the permission layer of the shared FastMCP service checks.
Inside this repo the boundary that actually holds is `ALLOWED_TOOLS` on each
agent: `dispatch()` refuses a name outside it, and `verify()` fails a run that
called one.

## Hotel tools

- Availability search is async: it polls until hotels are priced and returns only
  priced ones. When no city matches the query, it reports not-found rather than
  searching a stray hotel result.
- Sorting uses `SearchSortOption`: field is `PRICE`, `RATING` or `RECOMMENDED`
  (`RATING` is the star rating — there is no `STARS` value), order is lowercase
  `asc`/`desc`. `build_sort()` rejects anything else before the call is made.
- Filters: `minPrice`/`maxPrice` (total), `minStars`/`maxStars`, and `amenities`
  (a hotel must have all of them). They apply to the fields a search result
  carries. `getSearchResults` also takes a server-side `filters` argument, but its
  input type is not confirmed by the backend yet, so filtering runs in the tool.
- A search result is richer than it looked. `SearchResult.hotels` is
  `[seachBasicInfo]`, which carries `board`, `boardCode`, `roomName`,
  `cancelPolicy` (refundable plus dated penalties) and `optionRefId` — none of
  which the queries selected. Confirmed live: board `Breakfast Included` / code
  `1331`, penalties with amounts and deadlines, and an `optionRefId` that
  `refresh_hotel_price` accepts as-is. So `refundableOnly` and `mealPlan` are
  search filters, and a rate can be locked straight from a search result.
  Both search entry points now share one field selection, since `search` and
  `getSearchResults` return the same type and had drifted apart.
- The room options remain the way to see the *other* bookable choices at one
  hotel: `get_hotel_availability_options` (inside the search session, by `uuid`)
  or `get_hotel_options` (no session) take `refundableOnly`, `mealPlan` and
  price bounds.
- `getSearchHotelAvailability` returns `[RoomSearch]`, where each element holds
  `rooms` and `roomsOptions` side by side. `flatten_room_options()` pairs them
  into one record per bookable choice (room type + board + price + policy), which
  is what the tool returns and what the filters run on.
- `getHotelFullOptions` returns the same `[RoomSearch]` shape under `options`, so
  `get_hotel_options` flattens it the same way. Its `optionId` is the value
  `refresh_hotel_price` takes as `optionRefId`. On a room option that is the
  field name; on a search result the field is called `optionRefId` outright.
- `get_hotel_static_data` returns hotel content. On `HotelObj`, `medias` and
  `facilities` are `[String]`, so neither takes a subselection; `descriptions`
  and `phones` do. `street` is often null — the address is in `addressLines`.
  `extras=["descriptions", "facilities"]` adds the long text and the amenity
  list, opt-in because they are large.
- Paging: pass `pageNumber` (and `pageSize` on `get_hotel_search_results`); the
  result carries `count` and `hasMorePages`. Pass the same `checkIn`/`checkOut`
  as the search and `pricePerNight` comes back filled in — without them it is
  null on that path only, which left the agent dividing totals by hand.
- Currency/nationality default to org fallbacks when a request omits them; pass
  the real values per request, since nationality affects supplier pricing.
- Ops triage classifies each failure (supplier_timeout, validation_error,
  pnr_conflict, payment_failure, unknown), remembers seen signatures to avoid
  re-flagging, and tracks run_count per session.

`scripts/introspect.py` reads the live schema, so a query can be checked against
it instead of against memory. It marks which fields take a subselection, which is
the mistake the backend reports as `unexpected subselection set for non-object
field`:

    python scripts/introspect.py --type RoomSearch HotelObj
    python scripts/introspect.py --input HotelCriteriaSearchInput
    python scripts/introspect.py --query getSearch
    python scripts/introspect.py --dump          # scripts/schema_full.json

## Durable memory

`SessionMemory` in `runtime.py` is Layer 1 — per session, gone when the process
ends. `memory.py` is Layer 2: facts about a user or an organization that outlive
the session, on a temporal graph.

    python demo_memory.py

Four scenes, no infrastructure needed: a preference recalled in a new session; a
changed preference where the old fact is closed off rather than deleted; an org
default overridden by the user on the same key; an ops failure signature
recognised as recurring on the second session.

Two backends, one interface:

- `local` (default) — in-process graph. Deterministic, used by the demo and the
  tests. Supersede is driven by an explicit `key`; ranking is lexical with a
  recency fallback.
- `graphiti` — `graphiti_core` against FalkorDB, where entity resolution and
  hybrid search do the same job with an LLM and embeddings. Brought up by
  `docker compose up --build`.

Capture is explicit. The typed `add_user_episode` / `add_org_episode` /
`add_booking_episode` / `add_ops_episode` methods are the only way in — there is
no raw `add_episode()` pass-through, so supplier payloads and tool results cannot
reach memory on their own. Retrieval runs once per session, at the start, and is
cached on the agent context.

### Providers

Graphiti needs a chat model for extraction and an embedder for vectors.
OpenRouter serves chat completions only, so the embedder defaults to
`GRAPHITI_EMBEDDER=local` — a deterministic hashed bag-of-words vector in
`graphiti_embedder.py`, no key and no model download. That means the whole graph
runs on the single `OPENROUTER_API_KEY` you already have.

That embedder is lexical, not semantic, so wording that means the same thing in
different words scores lower than a trained model would give it. Retrieval is
hybrid (vector + BM25 + graph walk), so it still works. Set
`GRAPHITI_EMBEDDER=openai` with `GRAPHITI_EMBEDDER_API_KEY` for real embeddings.

### Docker

    docker compose up --build
    # console: http://localhost:8000   (press "Memory" in the header)
    # graph browser: http://localhost:3000
    docker compose run --rm app python demo_memory.py

That brings up FalkorDB and the service with `MEMORY_BACKEND=graphiti`, so memory
is durable in the graph and survives restarts. Your `.env` is read as-is and
never written to — compose only adds the container-side settings on top. If host
port 8000 is already taken:

    docker compose -f docker-compose.yml -f docker-compose.hostports.yml up --build

## Enrichment

The supplier feed knows stars, amenities and price. It does not know that guests
complain about the lifts, how far the hotel is from the Haram, that a wing has
been closed since spring, or what the weather will do during the stay.

Four tools cover that. `enrich_hotel_info` takes one hotel and answers on
reputation, location, facilities and risk. `enrich_destination` takes a city and
dates and answers on weather, travel advice and recent news.
`enrich_company_facts` takes a chain, brand owner or supplier;
`enrich_agency_facts` takes a travel agency.

### Where the answers come from

    open-meteo    weather, from a forecast API. No key, and numbers instead of prose.
    wikidata      company and agency facts, from a structured record rather than
                  from prose. No key. Every statement carries its own references,
                  which is what makes "only when verified" a rule and not a hope.
    gov-uk        travel advisories, from the government that issues them. No key.
                  Recognised by the safety gate on its host, like any other
                  government, rather than by an exception carved out for it.
    openrouter    the web plugin on the key already in use, for general lookups.
    playwright    one page, in a real browser, for sites that render nothing
                  without JavaScript. Off unless you install it.

Providers are asked in that order and the first useful answer wins. One that
fails is recorded and skipped rather than retried. Configure with
`WEB_SEARCH_BACKEND=openrouter` and `WEB_PLAYWRIGHT_ENABLED=true`; the two
keyless providers are on by default, and with nothing else set the remaining
domains say so instead of inventing.

### Company and agency facts

These two domains carry a fixed set of fields and nothing else:

    company_facts   legal_name, parent_company_or_owner, headquarters, founded,
                    official_website, ceo
    agency_facts    legal_or_trading_name, country, headquarters_or_address,
                    official_website, contact_phone, contact_email,
                    accreditation_or_licence

A closed schema is what makes "we do not carry that" a true statement rather
than a shrug. A source offering `annual_revenue` has it dropped and listed under
`fields_outside_schema`; a field nothing carried is listed under `missing`. Both
reach the answer as *not verified*, which is the point — the alternative is a
gap the model fills from its own recollection, indistinguishable from a fact.

Some fields may not be taken on a stranger's say-so. A CEO named by a travel
blog is a rumour, and a licence number quoted by a directory is worse, because a
customer could act on it. `ceo`, `contact_phone`, `contact_email` and
`accreditation_or_licence` are dropped unless the claim carries an authoritative
source, and named in `not_verified` rather than left silently absent.

Every claim carries an `authority` alongside its status, because counting how
many pages repeat a value says nothing about who published them:

    government        a state or recognised official authority
    entity_official   the subject's own website
    reference_backed  a structured record that cites its own source
    third_party       everything else, which is most of the web

Confirmed live: Accor's chief executive is a referenced Wikidata statement and is
reported; Hilton's and Marriott's carry no reference and come back as
`not_verified`. Hilton's `owned by` is *deprecated* on Wikidata, and reading it
would have named an owner that stopped being true in 2013 — deprecated
statements are skipped.

Once an `official_website` is confirmed, the subject's own pages count as
authoritative for the rest of that record. Finding the site and then not using it
would leave a fact quoted from the company's own leadership page ranking level
with a directory.

### An official advisory has to come from a government

An answer may call something an official government travel advisory only when a
government or recognised official authority published it. News sites, blogs,
aggregators and travel sites never satisfy that, however accurately they quote
it — the result reports `checks.official_advisory_verified`, and when it is
false the note says no official advisory was verified.

The host is read structurally: the last labels of the domain, against a list of
the suffixes states actually publish under. Searching for `gov` anywhere in the
string — which is how the tier ranking used to work — calls
`gov.uk.travel-deals.com` a government. That is a fine way to rank a page and a
bad way to decide whether a warning is official.

A gate needs something to gate. Until `gov-uk-advisory` existed, the advisory
domain had no provider at all unless `WEB_SEARCH_BACKEND` was set — which it is
not by default — so `enrich_destination` returned "no provider configured" and
every answer came out as "no official advisory verified". That was true only in
the sense that nobody had looked. Those two situations now read differently: a
domain with no provider reports `provider_configured: false` and says nothing
was searched, and deliberately does not set `official_advisory_verified` at all,
since asserting `false` would claim a search that never happened. The gate reads
absent as not-verified either way, so nothing is loosened by not asserting it.

Advisories are per country and `enrich_destination` is handed a city, so the
country comes from the same open-meteo geocoding the forecast already uses —
Muscat resolves to Oman, then to `/foreign-travel-advice/oman`. The handful of
countries GOV.UK spells differently are mapped from checked responses rather
than guessed (`United States` → `usa`, `Ivory Coast` → `cote-d-ivoire`), accents
are stripped first because geocoding answers "Côte d'Ivoire", and an unmapped
404 names the slug it tried so `countrySlug` can correct it instead of the
answer concluding the country has no advisory.

`verify()` enforces the same rule on the wording. An answer that calls a Reuters
report an official advisory fails; the same sentence passes when a `gov.uk`
source is behind it, whether it arrived from a fresh fetch or out of the index.
Naming the body is not the trigger — "Reuters reports that the Foreign Office
advises against travel" is the correctly attributed form, and an earlier version
of the check wrongly failed it.

### When the user says not to fetch

"using stored enrichment only", "do not fetch", "do not use fresh data", "use
existing enrichment only" — on a turn phrased that way, `search_enrichment` still
runs, because it only reads what an earlier fetch already stored. All four fetch
tools are refused *before* they run and are never recorded as tool calls, so the
restriction is a property of what the run actually did. The model is handed back
the reason and told which tool is still open, so the turn continues from the
index instead of stalling. If the entity, the date or the field is not there, the
answer says it is unavailable in stored enrichment and stops.

The flag is re-read every turn — a chat session reuses one context, so "do not
fetch" on turn 3 must not still be blocking on turn 4. A delegated child is the
exception: it keeps what its parent was told, since handing work on is not a way
round a restriction. `dispatch()` refuses as well, for anything reaching it
another way.

Detection is deliberately tight. It gates a hard refusal, so it holds only
phrasings that can mean one thing; "what does the cached data say" reads as a
question about the cache as easily as an instruction, and is left to the prompt.

Tight in both directions. The exclusivity word is required, and an explicit
permission to fetch is read first and wins, because a sentence often names the
restriction and then lifts it: "use stored enrichment first, then fetch if
missing" is an ordering, not a prohibition. Written with `only` optional, the
detector matched the bare phrase "use stored enrichment" and never reached the
clause that lifted it, so the fetch the user had asked for was refused. What
must not follow from that is a detector unable to refuse — "tell me if not
found" is an instruction about what to say, not permission to go and look.

A claim is kept only if its url appears in the search results the provider
returned. If nothing came back to check against, the claim is dropped — a url the
model wrote and no search returned is a guess wearing a citation.

Playwright is not in `requirements.txt` because it pulls a browser. Install it
when you want it:

    pip install -r requirements-playwright.txt
    playwright install chromium

It reads the page passed as `pageUrl` to `enrich_hotel_info`, and does nothing
without one.

### How a claim earns its place

Every claim keeps the page it came from; one whose source cannot be confirmed is
dropped rather than shown with a shrug. Then the evidence is counted, not scored:

    corroborated    two different sites say it
    single_source   one site says it, and the answer says so
    conflicting     two sites disagree, and both readings are kept

For the same field, agreement is decided on the numbers — "8.7 out of 10" and
"8.7 / 10" are one rating, 8.7 and 8.4 are a disagreement worth showing. There is
no confidence percentage anywhere in this, because there is no honest way to
compute one from a handful of pages.

Sources are ranked official, gov, maps, reviews, news, other, so a hotel's own
page outranks an aggregator when they differ.

Price, availability, board and cancellation come from the supplier alone. A
money-shaped value is dropped as it is parsed, and `verify` fails a run whose web
claims quote a price. Web text is treated as data throughout: instruction-shaped
phrasing is stripped before any model sees it.

Answers are cached for as long as they stay true — three hours for weather, a day
for advisories and risk, six hours for news, a week for reputation, a month for
location and facilities. Nothing runs on a timer; a caller asks, and the cache
decides whether the stored answer is still good.

## Searching what was already fetched

`enrich_hotel_info` and `enrich_destination` need the subject and the domain up
front. `search_enrichment` does not — it takes a plain question and reads
everything already fetched:

    search_enrichment("which of these has a pool problem")
    search_enrichment("what did we learn about Jeddah", entityType="city")
    search_enrichment("renovation", entityRef="Carawan Hotel")

Records are keyed the way the feed keys its snapshots — `entity_type` ("hotel" or
"city") and `entity_ref` — so a row here lines up with a snapshot there without a
translation step.

Every claim is embedded as it is written, inside the same call that fetched it,
so an embedding is never older than the claim it describes. One row per claim:
they arrive already split at a natural boundary — a field, a value, its sources —
so there is nothing to chunk, and splitting further would separate a fact from
its citation. Vectors come from the same hashed embedder the memory layer uses,
so the repo has one embedding scheme rather than two.

That embedder matches words, not meaning. "how warm will it be" shares nothing
with "29.2-34.5°C", so each domain carries a written list of the words people use
for it (`DOMAIN_WORDS`), added to both the stored fact and the question. It covers
the seven domains we own; a word nobody listed will still miss, and the test suite
says so out loud. Swapping in a real embedding model is a change to `embed_text`
alone.

Storage is SQLite with the similarity done in Python. A dedicated vector database
earns its keep at a scale this does not have — these are enrichment records for
the hotels and cities in play, not a document corpus — and SQLite adds nothing to
deploy, secure or back up. `VectorStore` is the seam to move behind pgvector the
day those numbers change. The file opens on first use, and falls back to memory
with a warning if the path is not writable.

### The score floor

Cosine over a bag of words is never exactly zero — an unrelated question still
shares `it`, `the`, a stray digit. Measured on real claims, a question inside a
domain scores 0.74–0.85 and one outside it scores 0.03–0.13, because `expand()`
puts a whole domain vocabulary on both sides of a genuine match. `MIN_SCORE = 0.35`
sits between the two bands, so `"is it muggy"` returns nothing and says so, instead
of returning a weather claim with a real citation attached to a question it does
not answer.

Pass `minScore=0` to see what was filtered.

## Agent context

Each agent runs on its own `AgentContext`. When one agent needs another, it does
not share that context:

    handover = await delegate(OpsTriageAgent(), brief, llm, parent_ctx)

The child gets the customer and the durable memory, because those belong to the
person rather than to the conversation. It does not get the parent's session
keys, tool results or retrieved facts — it builds its own prompt from the brief
and answers that. What comes back is a `Handover`: the answer, which tools were
used, and whether verification passed. None of the child's messages or tool
payloads reach the caller's history.

## Models

The chat page offers whichever models are configured, default first:

    OPENROUTER_MODEL / OPENROUTER_API_KEY        anthropic/claude-haiku-4.5
    OPENROUTER_MODEL_B / OPENROUTER_API_KEY_B    poolside/laguna-xs-2.1
    OPENROUTER_MODEL_C / OPENROUTER_API_KEY_C    google/gemma-4-31b-it

Each carries its own key because they are separate accounts, and a request is
paid for by the key belonging to the model it names — an unknown name falls back
to the default rather than borrowing someone else's credit. Whether a model is
listed with or without a `:free` suffix differs by account, so a name rejected as
unknown is retried once with the other spelling. A 402 that names an affordable
budget is retried once inside it.

Switching model in the page starts a fresh conversation, since the transcript
belongs to the model that produced it.

## The console, and what each panel is evidence of

`uvicorn api:app` then open the page. Three panels sit in the top bar, and each
exists because a claim about this service is otherwise only assertable in prose.

**Memory** — `GET /memory`. The durable facts for the user, current and
superseded, plus the context string injected at session start.

**Enrichment** — `GET /enrichment?q=`. The index behind `search_enrichment`,
queried directly. Every row carries its match score, its `status`
(`corroborated` / `single_source` / `conflicting`), the entity it belongs to, and
a link to the source. It never fetches: this is the index being read, not the web.
An empty `q` returns just `indexed_claims`, so the panel can show a count before
anyone searches.

**Delegate** — `POST /delegate`. Hands a brief to the ops agent on behalf of the
current session. Triggered by a person: neither agent has a delegation tool. The
response carries `parent_before` and `parent_after` — the caller's `tool_calls`
and session keys either side of the run — plus `parent_unchanged`:

    handover.agent      : ops_triage_agent
    handover.tools_used : ['get_queue_summary']       # names, never payloads
    parent_before       : {'tool_calls': ['remember_preference'], 'session_keys': []}
    parent_after        : {'tool_calls': ['remember_preference'], 'session_keys': []}
    parent_unchanged    : True

### `tools_called` is the turn, not the session

`ctx.tool_calls` accumulates for the life of a session on purpose — `verify()`
must still see a write tool called on turn 1 when it runs on turn 5. The page
needs the opposite, so `/chat` returns both: `tools_called` for the turn and
`tools_called_session` for everything so far.

This matters for the retrieval demo. Fetch first, then ask a question that should
be answered from the index: without the split, the second turn's chip row shows
`enrich_destination` alongside `search_enrichment` and appears to contradict the
claim being made.

### One turn at a time

A session runs one turn at a time, and Send and Delegate share the lock. Two
concurrent `/chat` calls would both read the transcript as it was before either
started and the later writer would drop the earlier turn; a `/delegate` snapshot
taken around a chat still in flight would straddle that chat's own tool calls and
report the parent as changed with no isolation failure behind it.

## Tests

    pytest -q                                   496 tests, no network, no side files
    docker compose exec app pytest -q           the same suite inside the image

The suite is hermetic: `tests/conftest.py` swaps the enrichment index and its
providers for in-process ones, so no test reaches open-meteo or writes
`enrichment_index.sqlite3` next to the source.

`tests/test_graphiti_backend.py` covers the Graphiti path with `graphiti_core`
stubbed — the group-id charset, the empty-query listing and the per-group graph,
which is where every production bug on that path came from.
`tests/test_web_enrich.py` covers provider fallback, corroboration, disagreement,
injected instructions and the money guard.
`tests/test_enrichment_domains.py` covers the two fact domains, the verified-only
fields, the closed schema, freshness, and which hosts count as a government —
including the ones that only look official.
`tests/test_verify_grounding.py` covers what the *answer* may say: the official
advisory label, and fields the fetch reported it could not confirm.
`tests/test_stored_only_routing.py` drives the real agent loop and asserts that
no fetch tool runs on a no-fetch turn.
`tests/test_delegation_isolation.py` is adversarial: every secret is a canary
string, the child is asked outright to reveal the parent, and the assertions read
what was actually sent to the child's model rather than what its context holds.
`tests/test_fetch_on_miss_and_advisory_discovery.py` pins two live discrepancies
by the prompts that found them: a fetch-permitting instruction read as a refusal,
and an advisory domain with no provider to search.

Live checks are gated one by one:

    RUN_LIVE=1 pytest tests/test_live.py        Yarvel and OpenRouter
    RUN_GRAPHITI=1 pytest tests/test_live.py    writes a fact into FalkorDB and reads it back

## Known limitations

`_SESSIONS` is never evicted; a session's `org_id` and `username` are fixed at
creation and ignored on later turns; `session_id` is client-supplied and
unauthenticated; and the enrichment index has no tenant column, so a claim
fetched in one session is retrievable from another. All four are fine for a
single-tenant dev console and none of them is fine for production.
