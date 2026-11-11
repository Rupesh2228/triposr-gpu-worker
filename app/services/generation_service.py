from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..config import Settings
from ..model import TripoSRModel


@dataclass
class Job:
    job_id: str
    image_path: Path
    status: str = "queued"
    error: str | None = None
    output_path: Path | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


class GenerationService:
    def __init__(self, settings: Settings, model: TripoSRModel) -> None:
        self.settings = settings
        self.model = model
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs)

    def submit(self, image: Image.Image) -> Job:
        job_id = str(uuid.uuid4())
        now = time.time()
        image_path = self.settings.storage_dir / f"{job_id}.input.png"
        output_path = self.settings.storage_dir / f"{job_id}.glb"
        image.save(image_path, format="PNG")
        job = Job(job_id, image_path, created_at=now, updated_at=now)
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run, job, output_path)
        return job

    def get(self, job_id: str) -> Job | None:
        self.cleanup()
        with self.lock:
            return self.jobs.get(job_id)

    def cleanup(self) -> None:
        cutoff = time.time() - self.settings.job_retention_seconds
        with self.lock:
            expired = [
                job_id
                for job_id, job in self.jobs.items()
                if job.updated_at < cutoff and job.status in {"completed", "failed"}
            ]
            for job_id in expired:
                job = self.jobs.pop(job_id)
                for path in (job.image_path, job.output_path):
                    if path:
                        path.unlink(missing_ok=True)

    def _run(self, job: Job, output_path: Path) -> None:
        self._update(job, status="running")
        try:
            from rembg import new_session
            from tsr.utils import remove_background, resize_foreground

            with Image.open(job.image_path).convert("RGBA") as source:
                image = remove_background(source, new_session())
            image = resize_foreground(image, self.settings.foreground_ratio)
            array = np.asarray(image).astype(np.float32) / 255.0
            rgb = array[:, :, :3] * array[:, :, 3:4] + (1 - array[:, :, 3:4]) * 0.5
            prepared = Image.fromarray((rgb * 255).astype(np.uint8))
            self.model.generate(prepared, output_path)
            job.output_path = output_path
            self._update(job, status="completed")
        except Exception as exc:
            self._update(job, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            job.image_path.unlink(missing_ok=True)

    def _update(self, job: Job, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(job, key, value)
            job.updated_at = time.time()

