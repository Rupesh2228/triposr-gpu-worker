from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["queued", "running", "completed", "failed", "expired"]
    error: str | None = None
    model_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    model_loaded: bool
    model_error: str | None = None
    gpu: dict[str, Any]

