"""OpenTelemetry tracing for the X (Twitter) MCP server.

Plan E.x Task 7B, Pattern B — server-side instrumentation. Each MCP
`tools/call` request emits an `mcp.<tool_name>` span; the resulting traces
land in Phoenix Cloud under ``service.name=x-mcp``.

Wire format notes (verified during the bridge's Plan E Task 0):
  * Phoenix Cloud's CF edge rejects ``application/json`` with HTTP 415.
    Only OTLP-over-protobuf is accepted, so we use
    ``opentelemetry-exporter-otlp-proto-http``.
  * Auth contract: ``authorization: Bearer <PHOENIX_API_KEY>`` (NOT
    ``api_key=…``) per Arize-ai/phoenix packages/phoenix-otel/otel.py:169.
  * Project routing: ``x-project-name: <project>`` (Arize 2026-05-08 release).

Context propagation: Python's contextvars give us automatic async propagation
across awaits — unlike the Cloudflare Workers side of the bridge, no
custom context manager registration is required. The OTel SDK's
``contextvars_context.ContextVarsRuntimeContext`` is the default.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
from typing import Any

from fastmcp.server.middleware import Middleware
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode, TraceFlags

logger = logging.getLogger(__name__)

SERVICE_NAME = "x-mcp"
SERVICE_VERSION = "0.1.15"
TRACER_NAME = "x-twitter-mcp-server"

_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")

_provider: TracerProvider | None = None


def _parse_otlp_headers(raw: str) -> dict[str, str]:
    """Parse OTEL_EXPORTER_OTLP_HEADERS (comma-separated key=value).

    Values may contain ``=`` (e.g. base64 in a Bearer token), so we split
    on the first ``=`` only.
    """
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out


def _parse_resource_attrs(raw: str | None) -> dict[str, str]:
    """Parse the OTel-standard OTEL_RESOURCE_ATTRIBUTES env var.

    Format: comma-separated ``key=value`` pairs. We mirror
    ``_parse_otlp_headers`` (first-``=`` split) so values containing ``=``
    survive unmangled.
    """
    if not raw:
        return {}
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out


def init_tracer_provider() -> TracerProvider:
    """Idempotent tracer-provider init.

    When ``OTEL_EXPORTER_OTLP_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_HEADERS`` is
    unset, returns a no-op provider so the MCP server still boots in local
    dev and on Fly machines before secrets land. Spans created against the
    no-op provider are silently dropped — zero overhead.

    Returns the singleton provider; safe to call repeatedly.
    """
    global _provider
    if _provider is not None:
        return _provider

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    environment = os.environ.get("ENVIRONMENT", "unknown")

    # Resource precedence: hardcoded defaults → OTEL_RESOURCE_ATTRIBUTES →
    # OTEL_SERVICE_NAME (highest). Reading env vars manually rather than
    # relying on Resource.create's detector aggregation because the SDK's
    # detector merge inverts what we want: passing attrs to Resource.create
    # makes them override env, but operators set env to override code defaults.
    attrs: dict[str, str] = {
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "deployment.environment": environment,
    }
    attrs.update(_parse_resource_attrs(os.environ.get("OTEL_RESOURCE_ATTRIBUTES")))
    env_service_name = os.environ.get("OTEL_SERVICE_NAME")
    if env_service_name:
        attrs["service.name"] = env_service_name

    resource = Resource(attrs)

    if not endpoint or not headers:
        logger.warning(
            "tracing: OTEL_EXPORTER_OTLP_{ENDPOINT,HEADERS} unset; spans dropped"
        )
        _provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(_provider)
        return _provider

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=_parse_otlp_headers(headers),
    )
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            max_queue_size=256,
            max_export_batch_size=32,
            schedule_delay_millis=1000,
            export_timeout_millis=5000,
        )
    )
    trace.set_tracer_provider(_provider)
    logger.info(
        "tracing: OTel provider initialized (service=%s env=%s endpoint=%s)",
        SERVICE_NAME,
        environment,
        endpoint,
    )
    return _provider


def shutdown_tracer_provider() -> None:
    """Flush and shut down the provider — called on process termination so
    in-flight spans land in Phoenix before Fly sends SIGKILL.
    """
    global _provider
    if _provider is None:
        return
    try:
        _provider.force_flush(timeout_millis=4500)
        _provider.shutdown()
    except Exception as exc:  # pragma: no cover - best effort on shutdown
        logger.warning("tracing: shutdown error (ignored): %s", exc)


def parse_traceparent(header: str | None) -> trace.SpanContext | None:
    """Parse a W3C traceparent header into a SpanContext.

    Returns ``None`` for malformed or missing input — the caller should fall
    back to the current active context.
    """
    if not header:
        return None
    match = _TRACEPARENT_RE.match(header)
    if not match:
        return None
    trace_id = int(match.group(1), 16)
    span_id = int(match.group(2), 16)
    flags = int(match.group(3), 16)
    return trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(flags),
    )


def restore_parent_context(traceparent: str | None) -> Context:
    """Restore an OTel Context from a traceparent header string.

    Returns the current active context when traceparent is missing or
    malformed — so a missing parent doesn't break the trace, it just makes
    the child span a synthetic root grouped by ``service.name=x-mcp``.
    """
    span_ctx = parse_traceparent(traceparent)
    if span_ctx is None:
        return otel_context.get_current()
    return trace.set_span_in_context(trace.NonRecordingSpan(span_ctx))


def extract_traceparent_from_meta(meta: Any) -> str | None:
    """Pull ``_otel_traceparent`` out of an MCP request's ``params._meta``.

    The MCP ``RequestParams.Meta`` model has ``extra="allow"``, so Pydantic
    parses ``{"_otel_traceparent": "..."}`` into the model but exposes it via
    ``model_extra`` or as a regular attribute on the parsed instance. We
    accept both shapes (Pydantic-model and bare dict) defensively because
    middleware sometimes sees pre-validation payloads.
    """
    if meta is None:
        return None
    # Pydantic v2 model — use model_extra, falling back to getattr for typed
    # fields that might be promoted to first-class in a future MCP schema.
    extras = getattr(meta, "model_extra", None)
    if isinstance(extras, dict):
        value = extras.get("_otel_traceparent")
        if isinstance(value, str):
            return value
    value = getattr(meta, "_otel_traceparent", None)
    if isinstance(value, str):
        return value
    if isinstance(meta, dict):
        value = meta.get("_otel_traceparent")
        if isinstance(value, str):
            return value
    return None


def get_tracer() -> trace.Tracer:
    """Return the project tracer. Triggers lazy init when called before the
    HTTP entrypoint (defensive — e.g. from stdio mode or a unit test)."""
    if _provider is None:
        init_tracer_provider()
    return trace.get_tracer(TRACER_NAME)


class TraceContextMiddleware:
    """ASGI middleware that restores an OTel parent context from the MCP JSON
    request body or the W3C ``traceparent`` HTTP header before FastMCP sees it.

    FastMCP 3.3.1 strips ``params._meta`` in its middleware layer (server.py
    line 955: CallToolRequestParams is reconstructed without _meta), so the
    only reliable injection point is the raw HTTP body, before FastMCP parses
    it.

    Priority:
      1. ``params._meta._otel_traceparent`` in JSON body (Worker injection)
      2. W3C ``traceparent`` HTTP request header (standard propagation)
      3. Active context — span becomes a synthetic root, no error
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        # Buffer the full request body once so we can inspect it AND replay
        # it for FastMCP, which reads it a second time over the ASGI channel.
        chunks: list[bytes] = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
        body = b"".join(chunks)

        traceparent: str | None = None

        # Try JSON body first (MCP injection by the bridge Worker).
        try:
            if body:
                payload = _json.loads(body)
                if isinstance(payload, dict):
                    params = payload.get("params")
                    if isinstance(params, dict):
                        meta = params.get("_meta")
                        if isinstance(meta, dict):
                            tp = meta.get("_otel_traceparent")
                            if isinstance(tp, str):
                                traceparent = tp
        except Exception:
            pass

        # Fall back to the W3C traceparent request header.
        if traceparent is None:
            for k, v in scope.get("headers", []):
                if k.lower() == b"traceparent":
                    traceparent = v.decode("ascii", errors="replace")
                    break

        parent_ctx = restore_parent_context(traceparent)
        token = otel_context.attach(parent_ctx)

        # Replay the buffered body to the downstream ASGI app.
        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        try:
            await self.app(scope, replay_receive, send)
        finally:
            otel_context.detach(token)


class TracingMiddleware(Middleware):
    """FastMCP middleware that opens an ``mcp.<tool_name>`` span around each
    ``tools/call``. Parent context is picked up from the active OTel context,
    which ``TraceContextMiddleware`` sets at the HTTP layer by restoring the
    traceparent injected into ``params._meta._otel_traceparent``.
    """

    # on_call_tool is invoked for `tools/call` requests; everything else
    # passes through via the Middleware base class dispatch.
    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        params = getattr(context, "message", None)
        tool_name = getattr(params, "name", "unknown") if params is not None else "unknown"

        parent_ctx = otel_context.get_current()
        tracer = get_tracer()
        span = tracer.start_span(
            f"mcp.{tool_name}",
            context=parent_ctx,
            kind=SpanKind.SERVER,
            attributes={
                "mcp.tool_name": tool_name,
                "gen_ai.system": "x-twitter",
            },
        )
        active_ctx = trace.set_span_in_context(span, parent_ctx)
        token = otel_context.attach(active_ctx)
        try:
            result = await call_next(context)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            otel_context.detach(token)
            span.end()

    # Pass-through for every other middleware hook so unhandled message
    # types reach `call_next` unmodified. FastMCP's Middleware base provides
    # these as no-ops; defining them locally would shadow that — leave the
    # base class to handle them via __call__'s _dispatch_handler.

    async def on_message(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_request(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_notification(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_initialize(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_list_tools(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_read_resource(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_get_prompt(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_list_resources(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_list_resource_templates(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)

    async def on_list_prompts(self, context: Any, call_next: Any) -> Any:
        return await call_next(context)
