"""Unit tests for link-entity field passthrough (entities / note_tweet / article).

Regression: get_bookmarks and get_tweet_details requested only
``id,text,created_at,author_id``, so payloads carried bare ``t.co`` tokens in
``text`` with no ``entities.urls[].expanded_url``. Downstream consumers (the
x-bookmark-scan Routine) were forced to resolve t.co over HTTP themselves,
which 403s from CCR datacenter egress IPs (t.co IP-blocks bots; UA is
irrelevant). Requesting ``entities``, ``note_tweet``, and ``article`` gives
consumers the pre-resolved final URLs (and full long-post text + X-Article
metadata) straight from the API — no t.co hop anywhere.

NOTE: ``src.x_twitter_mcp.server`` is imported lazily inside the fixture, not
at module top level — same reason as test_get_bookmarks.py (importing the
server claims OpenTelemetry's once-per-process global tracer provider, which
test_tracing relies on winning at collection time).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# The exact field set both tools must request. One constant so the two tests
# can't drift apart from each other.
EXPECTED_TWEET_FIELDS = {
    "id", "text", "created_at", "author_id", "entities", "note_tweet", "article",
}


class _StubSession:
    """Avoids the network calls _OAuth2Session makes in __init__."""

    user_id = "999"
    headers = {"Authorization": "Bearer test"}


@pytest.fixture
def server(monkeypatch):
    from src.x_twitter_mcp import server as srv

    monkeypatch.setattr(srv, "_OAuth2Session", lambda: _StubSession())
    monkeypatch.setattr(srv, "check_rate_limit", lambda _action: True)
    return srv


@pytest.mark.asyncio
async def test_get_bookmarks_requests_link_entity_fields(server):
    captured: dict[str, Any] = {}

    def _fake_request(method, session, tweet_id=None, params=None):
        captured["params"] = params
        return {"data": [], "meta": {}}

    with patch.object(server, "_bookmarks_request", side_effect=_fake_request):
        await server.get_bookmarks(count=10)

    requested = set(captured["params"]["tweet.fields"].split(","))
    assert EXPECTED_TWEET_FIELDS <= requested


@pytest.mark.asyncio
async def test_get_bookmarks_passes_entity_payload_through(server):
    bookmark = {
        "id": "1",
        "text": "look at this https://t.co/abc",
        "author_id": "42",
        "entities": {
            "urls": [{
                "url": "https://t.co/abc",
                "expanded_url": "https://example.com/post",
                "unwound_url": "https://example.com/post",
            }]
        },
        "note_tweet": {
            "text": "the full long-form text",
            "entities": {"urls": [{"url": "https://t.co/def",
                                   "expanded_url": "https://example.com/long"}]},
        },
        "article": {"title": "An X Article"},
    }
    payload = {"data": [bookmark], "meta": {}}
    with patch.object(server, "_bookmarks_request", return_value=payload):
        result = await server.get_bookmarks(count=10)

    assert result["bookmarks"] == [bookmark]  # nothing stripped


@pytest.mark.asyncio
async def test_get_tweet_details_requests_link_entity_fields(server, monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeResponse:
        data = None

    class _FakeClient:
        def get_tweet(self, id, tweet_fields=None, **kwargs):
            captured["tweet_fields"] = tweet_fields
            return _FakeResponse()

    monkeypatch.setattr(
        server, "initialize_twitter_clients", lambda: (_FakeClient(), None)
    )
    await server.get_tweet_details(tweet_id="123")

    assert EXPECTED_TWEET_FIELDS <= set(captured["tweet_fields"])
