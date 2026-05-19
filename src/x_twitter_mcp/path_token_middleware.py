"""Path-token auth middleware for x-twitter-mcp-server.

Validates that the first segment of the request path matches the
MCP_ACCESS_TOKEN env var. Strips the token from the path before
forwarding to the underlying ASGI app, so downstream routes still
match (/mcp, /sse).

Added by Dreamware Development for Fly.io deployment. Not part of
the upstream rafaljanicki/x-twitter-mcp-server distribution.
"""

import hmac
import os
from typing import Any, Awaitable, Callable

from starlette.responses import PlainTextResponse


Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


class PathTokenMiddleware:
    """ASGI middleware that requires the first path segment to be a secret token.

    URL shape expected by claude.ai custom connector:
        https://<host>/<MCP_ACCESS_TOKEN>/mcp

    Requests without a matching first segment receive 401 Unauthorized
    and never reach the underlying MCP server.
    """

    def __init__(self, app: Any, token: str) -> None:
        if not token:
            raise RuntimeError(
                "PathTokenMiddleware requires a non-empty token. "
                "Set MCP_ACCESS_TOKEN to a long random string (e.g. `openssl rand -hex 32`)."
            )
        self.app = app
        self._token = token

    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        parts = path.lstrip("/").split("/", 1)
        first_segment = parts[0] if parts else ""

        if not first_segment or not hmac.compare_digest(first_segment, self._token):
            response = PlainTextResponse("Unauthorized", status_code=401)
            await response(scope, receive, send)
            return

        # Rewrite scope to drop the token from path + raw_path.
        new_path = "/" + (parts[1] if len(parts) > 1 else "")
        new_scope = {
            **scope,
            "path": new_path,
            "raw_path": new_path.encode("utf-8"),
        }
        await self.app(new_scope, receive, send)


def from_env() -> "Callable[[Any], PathTokenMiddleware]":
    """Factory that reads MCP_ACCESS_TOKEN at app-build time.

    Raises at startup if the token is missing — the server refuses to
    boot rather than expose tools without auth.
    """
    token = os.environ.get("MCP_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "MCP_ACCESS_TOKEN env var is required. Set it via `fly secrets set MCP_ACCESS_TOKEN=$(openssl rand -hex 32)`."
        )
    return lambda app: PathTokenMiddleware(app, token)
