import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from config import Settings as _Settings

# Windows defaults to the Proactor loop, whose pipe transports are closed by the
# garbage collector after the loop has gone. That prints a RuntimeError traceback
# per httpx client at the end of an otherwise green run. The selector loop has no
# such finaliser, and nothing here needs subprocesses.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def settings(**overrides) -> _Settings:
    """Settings for a test. `_env_file=None` keeps the developer's .env out of
    it — otherwise a value nobody set in the test still shows up."""
    return _Settings(_env_file=None, **overrides)
import hotel_tools as srv
from hotel_tools import RefreshFreshnessTracker


class Seq:
    """Wrap a list of responses to return one per successive call of an op."""
    def __init__(self, items):
        self.items = list(items)


class FakeClient:
    """Returns a canned payload per operation_name. A value may be an Exception
    (raised), a Seq (one item per call), or a plain value."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def request(self, *, query, variables, operation_name, token=None, extra_headers=None):
        self.calls.append({"op": operation_name, "variables": variables, "token": token,
                           "query": query})
        val = self.responses.get(operation_name)
        if isinstance(val, Seq):
            return val.items.pop(0) if len(val.items) > 1 else val.items[0]
        if isinstance(val, Exception):
            raise val
        return val


@pytest.fixture
def fake_hasura(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(srv, "_client", client)
    return client


@pytest.fixture
def fresh_tracker(monkeypatch):
    tracker = RefreshFreshnessTracker()
    monkeypatch.setattr(srv, "refresh_tracker", tracker)
    return tracker


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(*a, **k):
        return None
    monkeypatch.setattr(srv.asyncio, "sleep", _instant)


@pytest.fixture(autouse=True)
def _hermetic_enrichment(monkeypatch):
    """Keep the suite off the network and out of the developer's index file.

    `web_tools` builds its enricher and index at import time from the real
    settings, so any test that drives a tool through the agent or the API would
    otherwise call open-meteo for real and write the answers into
    `enrichment_index.sqlite3` next to the source. Both are swapped for
    in-process equivalents; a test that wants providers builds its own Enricher.
    """
    import web_tools
    from enrichment_index import EnrichmentIndex, SqliteVectorStore
    from web_enrich import Cache, Enricher

    index = EnrichmentIndex(SqliteVectorStore(":memory:"))
    monkeypatch.setattr(web_tools, "_index", index)
    monkeypatch.setattr(web_tools, "_enricher", Enricher([], Cache(), index=index))


def envelope(status="success", message=None, data="{}"):
    return {"status": status, "message": message, "data": data}
