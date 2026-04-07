"""
SentimentScribe — FastAPI authentication dependencies.

Two auth methods:
  1. Supabase JWT — for users logged in via the Streamlit dashboard
  2. API Key (X-API-Key header) — for developers / programmatic access

Usage in endpoints:
    # Require API key
    @app.post("/batch")
    async def batch(req: ..., user: CurrentUser = Depends(require_api_key)):

    # Optional auth (does not reject unauthenticated requests)
    @app.post("/predict")
    async def predict(req: ..., user: Optional[CurrentUser] = Depends(optional_auth)):
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import NamedTuple, Optional

import httpx
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")


class CurrentUser(NamedTuple):
    user_id: str
    auth_method: str  # "jwt" | "api_key"


# ── JWT verification (Supabase-issued tokens) ─────────────────────────────────

def _verify_jwt(token: str) -> str:
    """Decode a Supabase JWT and return the user UUID (sub claim).

    Raises HTTPException 401 on failure.
    """
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SUPABASE_JWT_SECRET not configured on this server.",
        )
    try:
        from jose import jwt, JWTError

        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise ValueError("Missing sub claim")
        return user_id
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── API key verification ──────────────────────────────────────────────────────

async def _verify_api_key(raw_key: str) -> str:
    """Hash the raw key and look it up in the Supabase api_keys table.

    Returns the owner's user_id on success.
    Raises HTTPException 401/503 on failure.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase not configured on this server.",
        )

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    url = f"{SUPABASE_URL}/rest/v1/api_keys"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    params = {"key_hash": f"eq.{key_hash}", "select": "user_id,id"}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            rows = resp.json()
    except Exception as exc:
        logger.error("Supabase key lookup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify API key.",
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    # Best-effort: update last_used_at (fire-and-forget, don't block response)
    _update_last_used(rows[0]["id"])

    return rows[0]["user_id"]


def _update_last_used(key_id: str) -> None:
    """Fire-and-forget update of last_used_at — failures are silently ignored."""
    import asyncio
    from datetime import datetime, timezone

    async def _do():
        url = f"{SUPABASE_URL}/rest/v1/api_keys?id=eq.{key_id}"
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        body = {"last_used_at": datetime.now(timezone.utc).isoformat()}
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                await client.patch(url, headers=headers, json=body)
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_do())
    except Exception:
        pass


# ── FastAPI Depends ───────────────────────────────────────────────────────────

async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    """Require either an X-API-Key header or a valid Bearer JWT."""
    if x_api_key:
        user_id = await _verify_api_key(x_api_key)
        return CurrentUser(user_id=user_id, auth_method="api_key")

    if credentials:
        user_id = _verify_jwt(credentials.credentials)
        return CurrentUser(user_id=user_id, auth_method="jwt")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def optional_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[CurrentUser]:
    """Optional auth — returns CurrentUser if credentials provided, else None."""
    if x_api_key:
        try:
            user_id = await _verify_api_key(x_api_key)
            return CurrentUser(user_id=user_id, auth_method="api_key")
        except HTTPException:
            return None

    if credentials:
        try:
            user_id = _verify_jwt(credentials.credentials)
            return CurrentUser(user_id=user_id, auth_method="jwt")
        except HTTPException:
            return None

    return None
