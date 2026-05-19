"""OAuth 2.0 user-token refresh for the Twitter/X bookmarks endpoints.

Twitter access tokens issued via the OAuth 2.0 PKCE flow expire after
~2 hours. The refresh token (granted by the `offline.access` scope) is
single-use: each refresh response includes a new refresh_token and the
old one is revoked.

This machine runs with `auto_stop_machines='stop'`, so in-memory state
is lost between requests. Rotated refresh tokens are persisted to a
Fly volume at `OAUTH2_STATE_DIR` (defaults to `/data`). The first
refresh after a deploy uses the `TWITTER_OAUTH2_USER_REFRESH_TOKEN`
env var as bootstrap; subsequent refreshes read from the volume.

Added by Dreamware Development. Not part of the upstream
rafaljanicki/x-twitter-mcp-server distribution.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import requests


logger = logging.getLogger(__name__)

_TOKEN_ENDPOINT = "https://api.twitter.com/2/oauth2/token"
_REQUEST_TIMEOUT = 30
_EXPIRY_SKEW_SECONDS = 60

_STATE_DIR = Path(os.environ.get("OAUTH2_STATE_DIR", "/data"))
_REFRESH_TOKEN_FILE = _STATE_DIR / "twitter_oauth2_refresh_token"
_ACCESS_TOKEN_CACHE = _STATE_DIR / "twitter_oauth2_access_token.json"

_lock = Lock()


def _load_refresh_token() -> str:
    if _REFRESH_TOKEN_FILE.exists():
        try:
            token = _REFRESH_TOKEN_FILE.read_text().strip()
            if token:
                return token
        except OSError as exc:
            logger.warning("Failed reading persisted refresh token (%s); falling back to env", exc)

    env_token = os.environ.get("TWITTER_OAUTH2_USER_REFRESH_TOKEN", "").strip()
    if not env_token:
        raise EnvironmentError(
            "No OAuth 2.0 refresh token available. Set the Fly secret "
            "TWITTER_OAUTH2_USER_REFRESH_TOKEN as a bootstrap, and ensure "
            f"{_STATE_DIR} is a mounted Fly volume so rotated tokens persist."
        )
    return env_token


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _persist_refresh_token(token: str) -> None:
    try:
        _atomic_write(_REFRESH_TOKEN_FILE, token)
        logger.info("Persisted rotated refresh token to %s", _REFRESH_TOKEN_FILE)
    except OSError as exc:
        logger.error(
            "FAILED to persist rotated refresh token to %s: %s. "
            "The next cold start will read the stale bootstrap secret and "
            "the refresh will fail with invalid_grant. Update the Fly secret "
            "TWITTER_OAUTH2_USER_REFRESH_TOKEN manually to recover.",
            _REFRESH_TOKEN_FILE,
            exc,
        )


def _load_cached_access_token() -> Optional[str]:
    if not _ACCESS_TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(_ACCESS_TOKEN_CACHE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cached access token unreadable (%s); will refresh", exc)
        return None
    token = data.get("access_token")
    expires_at = float(data.get("expires_at", 0))
    if not token or expires_at <= time.time() + _EXPIRY_SKEW_SECONDS:
        return None
    return token


def _persist_access_token(token: str, expires_at: float) -> None:
    try:
        _atomic_write(
            _ACCESS_TOKEN_CACHE,
            json.dumps({"access_token": token, "expires_at": expires_at}),
        )
    except OSError as exc:
        logger.warning("Failed to cache access token (%s); refresh will repeat next request", exc)


def _do_refresh(refresh_token: str) -> tuple[str, str, float]:
    client_id = os.environ.get("TWITTER_OAUTH2_CLIENT_ID")
    if not client_id:
        raise EnvironmentError(
            "TWITTER_OAUTH2_CLIENT_ID is required to refresh the OAuth 2.0 user token. "
            "Set it to your Twitter app's OAuth 2.0 Client ID (Developer Portal → "
            "your app → Keys and tokens → OAuth 2.0 Client ID and Client Secret)."
        )
    client_secret = os.environ.get("TWITTER_OAUTH2_CLIENT_SECRET")

    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "client_id": client_id,
    }
    auth = (client_id, client_secret) if client_secret else None

    resp = requests.post(
        _TOKEN_ENDPOINT,
        data=data,
        auth=auth,
        timeout=_REQUEST_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Twitter OAuth 2.0 refresh failed ({resp.status_code}): {resp.text}"
        )

    payload = resp.json()
    access_token = payload["access_token"]
    new_refresh_token = payload.get("refresh_token", refresh_token)
    expires_in = float(payload.get("expires_in", 7200))
    expires_at = time.time() + expires_in
    return access_token, new_refresh_token, expires_at


def get_access_token(force_refresh: bool = False) -> str:
    """Return a valid OAuth 2.0 user access token, refreshing if needed.

    Args:
        force_refresh: If True, ignore the cached access token and refresh
            unconditionally. Used after a 401 from a bookmarks API call to
            cover the case where Twitter invalidated the token early.
    """
    with _lock:
        if not force_refresh:
            cached = _load_cached_access_token()
            if cached:
                return cached

        refresh_token = _load_refresh_token()
        access_token, new_refresh_token, expires_at = _do_refresh(refresh_token)

        if new_refresh_token != refresh_token:
            _persist_refresh_token(new_refresh_token)
        _persist_access_token(access_token, expires_at)
        logger.info(
            "Refreshed OAuth 2.0 access token (expires in %.0fs)",
            expires_at - time.time(),
        )
        return access_token
