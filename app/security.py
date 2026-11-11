from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    from .config_runtime import get_settings

    supplied = x_api_key
    if supplied is None and authorization and authorization.startswith("Bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, get_settings().api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
