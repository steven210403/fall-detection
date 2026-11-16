"""Health-check contracts (thesis plan §17.2).

Liveness never depends on the camera. Readiness reports model and local
dependency state separately.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CheckStatus = Literal["ok", "degraded", "unavailable"]


class ReadinessCheck(BaseModel):
    """One readiness probe result."""

    name: str
    status: CheckStatus
    detail: str | None = None


class HealthLive(BaseModel):
    """Process liveness payload."""

    status: Literal["ok"] = "ok"
    service: str
    version: str


class HealthReady(BaseModel):
    """Component readiness payload."""

    status: CheckStatus
    checks: list[ReadinessCheck] = Field(default_factory=list)


__all__ = ["CheckStatus", "HealthLive", "HealthReady", "ReadinessCheck"]
