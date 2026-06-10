"""Unit tests for get_bookmarks pagination-cursor surfacing.

Regression: get_bookmarks previously returned ``data["data"]`` only, discarding
``data["meta"]["next_token"]`` — the pagination cursor. Without the cursor the
caller can never page past the first ~100 most-recently-bookmarked tweets, so
historical bookmarks on deeper pages are unreachable. These tests pin the tool
to return both the bookmark list AND the next cursor, and to forward a supplied
cursor back to the API as ``pagination_token``.

NOTE: ``src.x_twitter_mcp.server`` is imported lazily inside the fixture/tests,
not at module top level. Importing it runs ``init_tracer_provider()``, which
claims OpenTelemetry's once-per-process global tracer provider. ``test_tracing``
installs its own provider at module-import time and relies on winning that slot;
a top-level import here (this file sorts first alphabetically) would steal it and
break ``test_tracing``'s span-capture assertions. Deferring the import to run
time lets ``test_tracing``'s collection-time setup run first.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class _StubSession:
    """Avoids the network calls _OAuth2Session makes in __init__."""

    user_id = "999"
    headers = {"Authorization": "Bearer test"}


@pytest.fixture
def server(monkeypatch):
    from src.x_twitter_mcp import server as srv

    # _OAuth2Session() hits /2/users/me in __init__; stub it out.
    monkeypatch.setattr(srv, "_OAuth2Session", lambda: _StubSession())
    monkeypatch.setattr(srv, "check_rate_limit", lambda _action: True)
    return srv


@pytest.mark.asyncio
async def test_get_bookmarks_surfaces_next_cursor(server):
    payload = {
        "data": [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
        "meta": {"next_token": "CURSOR_PAGE_2"},
    }
    with patch.object(server, "_bookmarks_request", return_value=payload):
        result = await server.get_bookmarks(count=100)
    assert result["bookmarks"] == payload["data"]
    assert result["next_cursor"] == "CURSOR_PAGE_2"


@pytest.mark.asyncio
async def test_get_bookmarks_last_page_has_null_cursor(server):
    payload = {"data": [{"id": "3", "text": "c"}]}  # no meta → no next page
    with patch.object(server, "_bookmarks_request", return_value=payload):
        result = await server.get_bookmarks(count=100)
    assert result["bookmarks"] == payload["data"]
    assert result["next_cursor"] is None


@pytest.mark.asyncio
async def test_get_bookmarks_forwards_cursor_as_pagination_token(server):
    captured: dict[str, Any] = {}

    def _fake_request(method, session, tweet_id=None, params=None):
        captured["params"] = params
        return {"data": [], "meta": {}}

    with patch.object(server, "_bookmarks_request", side_effect=_fake_request):
        await server.get_bookmarks(count=50, cursor="CURSOR_PAGE_2")

    assert captured["params"]["pagination_token"] == "CURSOR_PAGE_2"
    assert captured["params"]["max_results"] == 50


@pytest.mark.asyncio
async def test_get_bookmarks_clamps_count_to_50(server):
    captured: dict[str, Any] = {}

    def _fake_request(method, session, tweet_id=None, params=None):
        captured["params"] = params
        return {"data": [], "meta": {}}

    with patch.object(server, "_bookmarks_request", side_effect=_fake_request):
        await server.get_bookmarks(count=100)

    assert captured["params"]["max_results"] == 50


@pytest.mark.asyncio
async def test_get_bookmarks_defaults_to_50(server):
    captured: dict[str, Any] = {}

    def _fake_request(method, session, tweet_id=None, params=None):
        captured["params"] = params
        return {"data": [], "meta": {}}

    with patch.object(server, "_bookmarks_request", side_effect=_fake_request):
        await server.get_bookmarks()

    assert captured["params"]["max_results"] == 50


@pytest.mark.asyncio
async def test_delete_all_bookmarks_fetches_pages_of_50(server):
    captured: dict[str, Any] = {}

    def _fake_request(method, session, tweet_id=None, params=None):
        if method == "GET":
            captured["params"] = params
        return {"data": [], "meta": {}}

    with patch.object(server, "_bookmarks_request", side_effect=_fake_request):
        await server.delete_all_bookmarks()

    assert captured["params"]["max_results"] == 50
