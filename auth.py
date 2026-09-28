"""Authentication & identity resolution for the agent backend.

Supports two pluggable modes (switch via environment variables, no code change):

1. Admin API key (existing behavior)
   - `X-API-Key: <key>` header, or `Authorization: Bearer <key>`.

2. Signed bearer token (OAuth / OIDC)
   - HS256 JWT verified with `AUTH_JWT_SECRET`, or
   - opaque token verified against `AUTH_INTROSPECTION_URL` (POST {"token": ...}).

Identity claims read from the token: `sub` (or `user_id`/`uid`) -> user id,
`email`, and `role` (or `user_role`).
"""

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional

from flask import request


def get_api_key() -> str:
    return os.getenv("RAG_API_KEY", "").strip()


def check_auth() -> bool:
    """Loose auth check that also allows same-origin browser requests."""
    key = get_api_key()
    if not key:
        return True

    client_key = request.headers.get("X-API-Key", "").strip()
    if client_key and client_key == key:
        return True

    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer ") and auth_header[7:].strip() == key:
        return True

    origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
    host = request.host
    if host and (f"//{host}" in origin or host in origin):
        return True

    return False


def require_admin_key() -> bool:
    """Require the configured API key for administrative endpoints."""
    key = get_api_key()
    if not key:
        return False

    client_key = request.headers.get("X-API-Key", "").strip()
    if client_key == key:
        return True

    auth_header = request.headers.get("Authorization", "").strip()
    return auth_header.startswith("Bearer ") and auth_header[7:].strip() == key


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _verify_jwt(token: str, secret: str) -> Optional[Dict[str, Any]]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg", "HS256") != "HS256":
            return None

        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = base64.urlsafe_b64encode(
            hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not hmac.compare_digest(sig_b64, expected):
            return None

        payload = json.loads(_b64url_decode(payload_b64))
        exp = payload.get("exp")
        if exp and time.time() > exp:
            return None

        user_id = payload.get("sub") or payload.get("user_id") or payload.get("uid")
        if not user_id:
            return None

        return {
            "user_id": str(user_id),
            "email": payload.get("email"),
            "role": payload.get("role") or payload.get("user_role") or "employee",
            "provider": "oauth",
        }
    except Exception:
        return None


def _introspect(token: str, url: str) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"token": token}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))

        if data.get("active", True) is False:
            return None

        user_id = data.get("sub") or data.get("user_id") or data.get("uid")
        if not user_id:
            return None

        return {
            "user_id": str(user_id),
            "email": data.get("email"),
            "role": data.get("role") or data.get("user_role") or "employee",
            "provider": "oauth",
        }
    except Exception:
        return None


def verify_bearer_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a signed bearer token and return an identity dict, or None."""
    secret = os.getenv("AUTH_JWT_SECRET", "").strip()
    if secret:
        return _verify_jwt(token, secret)

    introspect_url = os.getenv("AUTH_INTROSPECTION_URL", "").strip()
    if introspect_url:
        return _introspect(token, introspect_url)

    return None


def resolve_identity() -> Optional[Dict[str, Any]]:
    """Return the authenticated identity dict, or None if unauthenticated.

    Checks (in order): bearer token (OAuth), then admin API key.
    """
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer "):
        identity = verify_bearer_token(auth_header[7:].strip())
        if identity:
            return identity

    if require_admin_key():
        return {"user_id": "admin", "email": None, "role": "admin", "provider": "api-key"}

    return None
