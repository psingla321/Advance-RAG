"""Caller identity: who is asking, for which tenant, with which ACL groups.

Two modes, selected by ``AUTH_MODE``:

* ``header`` (default, local demo) — identity comes from ``X-Tenant-Id`` /
  ``X-User-Id`` / ``X-User-Groups``. Anyone can claim any identity, so never
  expose this mode beyond a developer machine.
* ``oidc`` (production) — a ``Bearer`` access token is validated against the
  issuer's JWKS (signature, ``exp``/``nbf``, ``iss`` and ``aud``), and tenant,
  user and groups are read from configurable claims. Defaults match Microsoft
  Entra ID (``tid``, ``oid``, ``roles``).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from fastapi import Header, HTTPException

from app.config import get_settings

log = logging.getLogger("nexus.auth")

IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_GROUPS = 20


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    user_id: str
    groups: list[str]

    @property
    def principal(self) -> str:
        return f"{self.tenant_id}:{self.user_id}"


def parse_groups(raw: str | list[str]) -> list[str]:
    items = raw.split(",") if isinstance(raw, str) else [str(g) for g in raw]
    groups = sorted({g.strip() for g in items if g.strip()})
    if not groups or len(groups) > MAX_GROUPS or not all(IDENTIFIER.match(g) for g in groups):
        raise HTTPException(400, f"Groups must be 1-{MAX_GROUPS} comma-separated identifiers ([A-Za-z0-9_-]).")
    return groups


@lru_cache
def _jwks_client(url: str) -> Any:
    from jwt import PyJWKClient

    return PyJWKClient(url, cache_keys=True, lifespan=3600)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def verify_token(token: str) -> dict[str, Any]:
    """Validate a bearer token and return its claims. Blocking: JWKS may be fetched."""
    import jwt

    settings = get_settings()
    if not (settings.oidc_jwks_url and settings.oidc_issuer and settings.oidc_audience):
        raise HTTPException(500, "AUTH_MODE=oidc requires OIDC_JWKS_URL, OIDC_ISSUER and OIDC_AUDIENCE.")
    try:
        key = _jwks_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token).key
        return jwt.decode(
            token, key, algorithms=settings.oidc_algorithms, audience=settings.oidc_audience,
            issuer=settings.oidc_issuer, options={"require": ["exp", "iss", "aud"]}, leeway=30,
        )
    except jwt.PyJWTError as exc:
        log.info("Rejected bearer token: %s", exc)
        raise _unauthorized("Invalid or expired access token.") from exc


def identity_from_claims(claims: dict[str, Any]) -> Identity:
    settings = get_settings()
    tenant = str(claims.get(settings.oidc_tenant_claim) or settings.oidc_default_tenant or "")
    user = str(claims.get(settings.oidc_user_claim) or claims.get("sub") or "")
    if not IDENTIFIER.match(tenant) or not IDENTIFIER.match(user):
        raise HTTPException(403, "The access token does not carry a usable tenant and user id.")
    raw_groups = claims.get(settings.oidc_groups_claim) or []
    if isinstance(raw_groups, str):
        raw_groups = raw_groups.split(",")
    # Every authenticated user is in "all"; unusable claim values are dropped rather than failing the call.
    groups = ["all", *[g for g in map(str, raw_groups) if IDENTIFIER.match(g)]][:MAX_GROUPS]
    return Identity(tenant, user, parse_groups(groups))


async def get_identity(
    authorization: str | None = Header(None),
    x_tenant_id: str = Header("demo"),
    x_user_groups: str = Header("all"),
    x_user_id: str = Header("anonymous"),
) -> Identity:
    if get_settings().auth_mode == "oidc":
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise _unauthorized("Missing bearer token.")
        return identity_from_claims(await asyncio.to_thread(verify_token, token))

    if not IDENTIFIER.match(x_tenant_id) or not IDENTIFIER.match(x_user_id):
        raise HTTPException(400, "Invalid tenant or user id.")
    return Identity(x_tenant_id, x_user_id, parse_groups(x_user_groups))
