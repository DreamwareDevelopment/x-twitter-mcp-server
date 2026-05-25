"""Unit tests for src/x_twitter_mcp/tracing.py.

Plan E.x Task 7B. We exercise the TracingMiddleware directly with a stub
context + call_next rather than spinning up the full FastMCP server (which
requires Twitter API creds at import time). The InMemorySpanExporter
captures every span emitted by `mcp.<tool_name>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

# Install the test provider as the global BEFORE importing the tracing
# module — OTel's SDK enforces `set_tracer_provider` once per process; any
# later call (e.g. tracing.init_tracer_provider's no-op fallback) is a
# silent no-op. Sharing one exporter across all tests is fine; we reset
# captured spans at the start of each test via the autouse fixture.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider(resource=Resource.create({"service.name": "x-mcp-test"}))
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)

from src.x_twitter_mcp.tracing import (  # noqa: E402  (must come after provider install)
    TraceContextMiddleware,
    TracingMiddleware,
    extract_traceparent_from_meta,
    parse_traceparent,
    restore_parent_context,
)

# Pin the tracing module's lazy-init singleton to our test provider so
# get_tracer()'s `if _provider is None` short-circuits cleanly. Without
# this, the module's first call would try to install a fresh provider and
# the SDK would warn (the global is already set, so the second install is
# a silent no-op — but the module's _provider stays None, repeating the
# warning each time).
import src.x_twitter_mcp.tracing as _tracing_mod  # noqa: E402

_tracing_mod._provider = _PROVIDER  # type: ignore[attr-defined]

VALID_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture(autouse=True)
def reset_exporter():
    """Clear captured spans before each test. Avoids cross-test bleed."""
    _EXPORTER.clear()
    yield _EXPORTER


def test_parse_traceparent_valid():
    ctx = parse_traceparent(VALID_TRACEPARENT)
    assert ctx is not None
    assert format(ctx.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(ctx.span_id, "016x") == "b7ad6b7169203331"
    assert ctx.is_remote is True


def test_parse_traceparent_invalid():
    assert parse_traceparent(None) is None
    assert parse_traceparent("") is None
    assert parse_traceparent("malformed") is None


def test_restore_parent_context_missing_returns_active():
    # No traceparent → restored ctx == current active (which is empty here).
    restored = restore_parent_context(None)
    assert restored is not None  # smoke: returns a Context, not None


@dataclass
class _StubMeta:
    """Mimics MCP RequestParams.Meta with `model_extra` for traceparent."""
    model_extra: dict[str, Any]


@dataclass
class _StubMessage:
    name: str
    arguments: dict[str, Any]
    meta: Any


@dataclass
class _StubContext:
    message: _StubMessage
    method: str = "tools/call"
    type: str = "request"


def _make_context(tool_name: str, traceparent: str | None) -> _StubContext:
    meta = None
    if traceparent is not None:
        meta = _StubMeta(model_extra={"_otel_traceparent": traceparent})
    return _StubContext(
        message=_StubMessage(name=tool_name, arguments={}, meta=meta),
    )


def test_extract_traceparent_from_pydantic_extra():
    meta = _StubMeta(model_extra={"_otel_traceparent": VALID_TRACEPARENT})
    assert extract_traceparent_from_meta(meta) == VALID_TRACEPARENT


def test_extract_traceparent_from_bare_dict():
    assert extract_traceparent_from_meta({"_otel_traceparent": VALID_TRACEPARENT}) == VALID_TRACEPARENT


def test_extract_traceparent_returns_none_for_missing():
    assert extract_traceparent_from_meta(None) is None
    assert extract_traceparent_from_meta(_StubMeta(model_extra={})) is None


@pytest.mark.asyncio
async def test_on_call_tool_emits_named_span(reset_exporter: InMemorySpanExporter):
    middleware = TracingMiddleware()
    ctx = _make_context("search_twitter", traceparent=None)
    sentinel = object()

    async def call_next(_: Any) -> Any:
        return sentinel

    result = await middleware.on_call_tool(ctx, call_next)
    assert result is sentinel

    spans = reset_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "mcp.search_twitter"
    assert span.attributes["mcp.tool_name"] == "search_twitter"
    assert span.attributes["gen_ai.system"] == "x-twitter"
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_on_call_tool_records_exception(reset_exporter: InMemorySpanExporter):
    middleware = TracingMiddleware()
    ctx = _make_context("post_tweet", traceparent=None)
    boom = RuntimeError("Twitter API 503")

    async def call_next(_: Any) -> Any:
        raise boom

    with pytest.raises(RuntimeError):
        await middleware.on_call_tool(ctx, call_next)

    spans = reset_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    # Exception is recorded as an event
    assert any(event.name == "exception" for event in span.events)


@pytest.mark.asyncio
async def test_on_call_tool_propagates_traceparent(reset_exporter: InMemorySpanExporter):
    middleware = TracingMiddleware()
    # TracingMiddleware now reads from otel_context.get_current(); the HTTP
    # layer (TraceContextMiddleware) is responsible for attaching the parent.
    ctx = _make_context("get_user_profile", traceparent=None)

    async def call_next(_: Any) -> Any:
        return None

    # Simulate what TraceContextMiddleware does at the HTTP layer.
    parent_ctx = restore_parent_context(VALID_TRACEPARENT)
    token = otel_context.attach(parent_ctx)
    try:
        await middleware.on_call_tool(ctx, call_next)
    finally:
        otel_context.detach(token)

    spans = reset_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Trace ID restored from the injected traceparent.
    assert format(span.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    # Parent span ID matches the traceparent's span ID.
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == "b7ad6b7169203331"


# ---------------------------------------------------------------------------
# TraceContextMiddleware — ASGI-layer traceparent extraction
# ---------------------------------------------------------------------------

import json  # noqa: E402


def _make_asgi_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": headers or [],
        "query_string": b"",
    }


def _body_receive(body: bytes):
    """Return an async receive callable that yields the given body once."""
    consumed = False

    async def receive():
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


@pytest.mark.asyncio
async def test_trace_context_middleware_extracts_from_body(
    reset_exporter: InMemorySpanExporter,
):
    """TraceContextMiddleware restores parent context from JSON body _meta."""
    captured_ctx: list[Any] = []

    async def inner_app(scope: Any, receive: Any, send: Any) -> None:
        captured_ctx.append(otel_context.get_current())
        # Consume the replayed body to confirm it is intact.
        msg = await receive()
        assert msg["body"] != b""

    mw = TraceContextMiddleware(inner_app)
    payload = {
        "method": "tools/call",
        "params": {
            "name": "search_twitter",
            "arguments": {},
            "_meta": {"_otel_traceparent": VALID_TRACEPARENT},
        },
    }
    body = json.dumps(payload).encode()
    scope = _make_asgi_scope()
    await mw(scope, _body_receive(body), None)

    assert len(captured_ctx) == 1
    span_ctx = trace.get_current_span(captured_ctx[0]).get_span_context()
    assert format(span_ctx.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(span_ctx.span_id, "016x") == "b7ad6b7169203331"


@pytest.mark.asyncio
async def test_trace_context_middleware_extracts_from_header(
    reset_exporter: InMemorySpanExporter,
):
    """TraceContextMiddleware falls back to W3C traceparent header."""
    captured_ctx: list[Any] = []

    async def inner_app(scope: Any, receive: Any, send: Any) -> None:
        captured_ctx.append(otel_context.get_current())

    mw = TraceContextMiddleware(inner_app)
    scope = _make_asgi_scope(
        headers=[(b"traceparent", VALID_TRACEPARENT.encode())]
    )
    await mw(scope, _body_receive(b"{}"), None)

    assert len(captured_ctx) == 1
    span_ctx = trace.get_current_span(captured_ctx[0]).get_span_context()
    assert format(span_ctx.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"


@pytest.mark.asyncio
async def test_trace_context_middleware_passthrough_non_http(
    reset_exporter: InMemorySpanExporter,
):
    """Non-HTTP scopes pass through without errors or OTel side effects."""
    called: list[bool] = []

    async def inner_app(scope: Any, receive: Any, send: Any) -> None:
        called.append(True)

    mw = TraceContextMiddleware(inner_app)
    scope = {"type": "websocket", "path": "/ws"}
    await mw(scope, _body_receive(b""), None)
    assert called == [True]
