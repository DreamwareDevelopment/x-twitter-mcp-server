"""Unit tests for the Tweepy response guards (SEC-873).

Both ``get_tweet_details`` and ``search_twitter`` were reported failing with
``'dict' object has no attribute 'data'``. That signature appears whenever a
result item arrives as a plain dict rather than a Tweepy model: the list
comprehensions did ``item.data`` unconditionally, so an unmodelled payload
raised an AttributeError several frames from its cause, with nothing in the
message a caller could act on.

These tests pin both halves of the fix:

* result items are read through ``_payload``, which accepts a Tweepy model or
  a raw dict, so the reported AttributeError can no longer occur; and
* a missing ``.data`` produces a ToolError carrying a ``[code=...]`` marker
  and Twitter's own ``errors`` detail, instead of an AttributeError, a
  TypeError, or a silent ``None``.

NOTE: ``src.x_twitter_mcp.server`` is imported lazily inside the fixture, not
at module top level, for the reason test_link_entity_fields.py documents
(importing the server claims OpenTelemetry's once-per-process global tracer
provider, which test_tracing relies on winning at collection time).
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError


class _Model:
    """Stands in for a Tweepy model (``Tweet``/``User``), which carries the raw
    payload dict on ``.data``."""

    def __init__(self, data: dict):
        self.data = data


class _Response:
    """Stands in for ``tweepy.client.Response``."""

    def __init__(self, data=None, errors=None):
        self.data = data
        self.errors = errors or []


@pytest.fixture
def server(monkeypatch):
    from src.x_twitter_mcp import server as srv

    monkeypatch.setattr(srv, "check_rate_limit", lambda _action: True)
    return srv


def _client(monkeypatch, server, **methods):
    """Install a fake Tweepy client exposing only the named methods."""
    fake = type("_FakeClient", (), methods)()
    monkeypatch.setattr(server, "initialize_twitter_clients", lambda: (fake, None))
    return fake


# --- the reported signature: unmodelled dict items ------------------------

@pytest.mark.asyncio
async def test_search_twitter_accepts_raw_dict_items(server, monkeypatch):
    """The SEC-873 signature: Tweepy left the items as plain dicts."""
    tweets = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    _client(
        monkeypatch, server,
        search_recent_tweets=lambda self, **kw: _Response(data=tweets),
    )

    result = await server.search_twitter(query="from:someone", count=10)

    assert result == tweets


@pytest.mark.asyncio
async def test_search_twitter_accepts_modelled_items(server, monkeypatch):
    """The ordinary path: Tweepy modelled the items, payload on `.data`."""
    tweets = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    _client(
        monkeypatch, server,
        search_recent_tweets=lambda self, **kw: _Response(
            data=[_Model(t) for t in tweets]
        ),
    )

    result = await server.search_twitter(query="from:someone", count=10)

    assert result == tweets


@pytest.mark.asyncio
async def test_search_twitter_empty_page_is_not_an_error(server, monkeypatch):
    """A search that matched nothing is a legitimate empty result, not a fault."""
    _client(
        monkeypatch, server,
        search_recent_tweets=lambda self, **kw: _Response(data=None),
    )

    assert await server.search_twitter(query="from:nobody", count=10) == []


def test_payload_rejects_an_unusable_item(server):
    with pytest.raises(ToolError) as exc:
        server._payload(object(), "tweets")

    assert "code=twitter_malformed_item" in str(exc.value)


# --- singular lookups: a missing `.data` is a described failure ------------

@pytest.mark.asyncio
async def test_get_tweet_details_returns_payload(server, monkeypatch):
    tweet = {"id": "123", "text": "hello", "author_id": "42"}
    _client(monkeypatch, server, get_tweet=lambda self, **kw: _Response(data=_Model(tweet)))

    assert await server.get_tweet_details(tweet_id="123") == tweet


@pytest.mark.asyncio
async def test_get_tweet_details_surfaces_twitter_error_detail(server, monkeypatch):
    """A deleted/suspended tweet: 200 with an errors body and no data."""
    _client(
        monkeypatch, server,
        get_tweet=lambda self, **kw: _Response(
            data=None,
            errors=[{"title": "Not Found Error",
                     "detail": "Could not find tweet with id: [123]."}],
        ),
    )

    with pytest.raises(ToolError) as exc:
        await server.get_tweet_details(tweet_id="123")

    message = str(exc.value)
    assert "code=twitter_no_data" in message
    assert "Could not find tweet with id: [123]." in message


@pytest.mark.asyncio
async def test_get_tweet_details_error_is_descriptive_without_errors_body(
    server, monkeypatch
):
    _client(monkeypatch, server, get_tweet=lambda self, **kw: _Response(data=None))

    with pytest.raises(ToolError) as exc:
        await server.get_tweet_details(tweet_id="123")

    assert "no detail supplied by Twitter" in str(exc.value)


@pytest.mark.asyncio
async def test_get_user_by_id_surfaces_missing_data(server, monkeypatch):
    _client(
        monkeypatch, server,
        get_user=lambda self, **kw: _Response(
            data=None, errors=[{"detail": "User has been suspended."}]
        ),
    )

    with pytest.raises(ToolError) as exc:
        await server.get_user_by_id(user_id="42")

    assert "code=twitter_no_data" in str(exc.value)
    assert "User has been suspended." in str(exc.value)


# --- mutations: acknowledgement flags -------------------------------------

@pytest.mark.asyncio
async def test_favorite_tweet_returns_flag(server, monkeypatch):
    _client(monkeypatch, server, like=lambda self, **kw: _Response(data={"liked": True}))

    assert await server.favorite_tweet(tweet_id="1") == {"tweet_id": "1", "liked": True}


@pytest.mark.asyncio
async def test_favorite_tweet_missing_data_is_a_tool_error(server, monkeypatch):
    """Previously a TypeError: 'NoneType' object is not subscriptable."""
    _client(monkeypatch, server, like=lambda self, **kw: _Response(data=None))

    with pytest.raises(ToolError) as exc:
        await server.favorite_tweet(tweet_id="1")

    assert "code=twitter_no_data" in str(exc.value)


@pytest.mark.asyncio
async def test_favorite_tweet_missing_flag_is_a_tool_error(server, monkeypatch):
    """Previously a KeyError with no indication of which field was absent."""
    _client(monkeypatch, server, like=lambda self, **kw: _Response(data={}))

    with pytest.raises(ToolError) as exc:
        await server.favorite_tweet(tweet_id="1")

    assert "code=twitter_malformed_response" in str(exc.value)
    assert "'liked'" in str(exc.value)


@pytest.mark.asyncio
async def test_unfavorite_tweet_inverts_the_flag(server, monkeypatch):
    _client(monkeypatch, server, unlike=lambda self, **kw: _Response(data={"liked": False}))

    assert await server.unfavorite_tweet(tweet_id="1") == {"tweet_id": "1", "liked": True}


# --- creates: a failed post is no longer a silent null --------------------

@pytest.mark.asyncio
async def test_post_tweet_failure_is_a_tool_error(server, monkeypatch):
    _client(
        monkeypatch, server,
        create_tweet=lambda self, **kw: _Response(
            data=None, errors=[{"detail": "Tweet needs to be a bit shorter."}]
        ),
    )

    with pytest.raises(ToolError) as exc:
        await server.post_tweet(text="x" * 500)

    assert "Tweet needs to be a bit shorter." in str(exc.value)


@pytest.mark.asyncio
async def test_post_tweet_returns_payload(server, monkeypatch):
    created = {"id": "9", "text": "hi"}
    _client(monkeypatch, server, create_tweet=lambda self, **kw: _Response(data=created))

    assert await server.post_tweet(text="hi") == created
