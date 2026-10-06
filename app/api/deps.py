"""Shared dependencies: operator authentication."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings


def require_operator(x_operator_token: str | None = Header(default=None)) -> str:
    """Gate anything that can change what the signal heads display.

    With no token configured the endpoint refuses rather than defaulting to
    open. An unauthenticated write path into a signal controller is not a
    development convenience.
    """
    settings = get_settings()
    if not settings.operator_token:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "operator actions are disabled: set OPERATOR_TOKEN to enable them",
        )
    if x_operator_token != settings.operator_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid operator token")
    return x_operator_token
