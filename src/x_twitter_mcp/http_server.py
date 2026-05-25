import atexit
import os
import signal
import sys
from typing import Any, Callable, Optional

import uvicorn
from starlette.middleware.cors import CORSMiddleware

from .server import server
from .middleware import SmitheryConfigMiddleware
from .path_token_middleware import from_env as path_token_from_env
from .tracing import TraceContextMiddleware, shutdown_tracer_provider


def _create_asgi_app() -> Any:
    """Create an ASGI app from the FastMCP server with broad compatibility.

    Tries multiple factory methods to account for FastMCP version differences.
    """
    app_factory: Optional[Callable[[], Any]] = None

    # Prefer streamable HTTP app if available
    if hasattr(server, "streamable_http_app") and callable(getattr(server, "streamable_http_app")):
        app_factory = getattr(server, "streamable_http_app")
    elif hasattr(server, "http_app") and callable(getattr(server, "http_app")):
        app_factory = getattr(server, "http_app")

    if app_factory is not None:
        app = app_factory()  # type: ignore[no-any-return]
    else:
        # Fall back to a prebuilt ASGI app attribute if present
        app = getattr(server, "asgi_app", None)
        if app is None:
            raise RuntimeError(
                "FastMCP server does not expose an HTTP ASGI app. "
                "Please upgrade fastmcp or expose http_app/streamable_http_app."
            )

    # CORS for browser-based MCP clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id", "mcp-protocol-version"],
        max_age=86400,
    )

    # Restore OTel parent context from params._meta._otel_traceparent (JSON
    # body) or W3C traceparent header before FastMCP strips _meta from
    # MiddlewareContext (FastMCP 3.3.1 server.py:955).
    app = TraceContextMiddleware(app)

    # Inject Smithery config-per-request and map to env vars used by Tweepy setup
    app = SmitheryConfigMiddleware(app)

    # Outermost gate: require MCP_ACCESS_TOKEN as the first path segment.
    # Refuses to start if the env var is missing — secure by default.
    app = path_token_from_env()(app)
    return app


# Uvicorn entrypoint expects an ASGI app at module level
app = _create_asgi_app()


def _on_shutdown(*_args: Any) -> None:
    """Graceful drain — flush in-flight OTel spans before Fly sends SIGKILL.

    Fly's `kill_signal = SIGINT` + `kill_timeout = 5s` gives us a 5s window
    after SIGINT to land spans in Phoenix. uvicorn's own signal handlers run
    first and start the graceful shutdown; this hook fires from atexit at
    the very end of the process lifetime.
    """
    shutdown_tracer_provider()


def main() -> None:
    """Run the ASGI server using Uvicorn.

    Smithery sets PORT; default to 8081 for local testing.
    """
    atexit.register(_on_shutdown)
    # Uvicorn installs its own SIGINT/SIGTERM handlers and exits cleanly;
    # atexit catches every termination path (including SIGTERM via Fly).
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()


