from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_key: str
    cors_origins: tuple[str, ...]
    max_upload_mb: int
    max_concurrent_jobs: int
    job_retention_seconds: int
    chunk_size: int
    mc_resolution: int
    foreground_ratio: float
    model_path: str
    triposr_repo_dir: Path
    storage_dir: Path
    load_model_on_startup: bool

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("WORKER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("WORKER_API_KEY must be configured")
        return cls(
            api_key=api_key,
            cors_origins=tuple(_csv(os.getenv("CORS_ORIGINS", ""))),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
            max_concurrent_jobs=1,
            job_retention_seconds=int(os.getenv("JOB_RETENTION_SECONDS", "3600")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "8192")),
            mc_resolution=int(os.getenv("MC_RESOLUTION", "256")),
            foreground_ratio=float(os.getenv("FOREGROUND_RATIO", "0.85")),
            model_path=os.getenv("TRIPOSR_MODEL", "stabilityai/TripoSR"),
            triposr_repo_dir=Path(
                os.getenv("TRIPOSR_REPO_DIR", "./third_party/TripoSR")
            ).resolve(),
            storage_dir=Path(os.getenv("WORKER_STORAGE_DIR", "./runtime")).resolve(),
            load_model_on_startup=os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower()
            in {"1", "true", "yes"},
        )

