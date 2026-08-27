import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from config import Settings as _Settings


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


def envelope(status="success", message=None, data="{}"):
    return {"status": status, "message": message, "data": data}
