from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config_runtime import get_settings
from .gpu import gpu_status
from .model import TripoSRModel
from .schemas import HealthResponse, JobResponse
from .security import require_api_key
from .services.generation_service import GenerationService
from .services.image_service import store_image

settings = get_settings()
model = TripoSRModel(settings)
generation = GenerationService(settings, model)
startup_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global startup_error
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if settings.load_model_on_startup:
        try:
            model.load()
        except Exception as exc:
            startup_error = f"{type(exc).__name__}: {exc}"
    yield
    generation.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="TripoSR GPU Worker", lifespan=lifespan)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

@app.get("/health", response_model=HealthResponse, dependencies=[Depends(require_api_key)])
def health() -> HealthResponse:
    gpu = gpu_status()
    return HealthResponse(
        status="ok" if model.loaded else "unavailable",
        cuda_available=gpu["available"],
        model_loaded=model.loaded,
        model_error=startup_error,
        gpu=gpu,
    )


@app.get("/gpu-status", dependencies=[Depends(require_api_key)])
def gpu() -> dict:
    return gpu_status()


@app.post("/generate", response_model=JobResponse, status_code=202, dependencies=[Depends(require_api_key)])
async def generate(image: UploadFile = File(...)) -> JobResponse:
    checked_image = await store_image(image, settings)
    if not model.loaded:
        raise HTTPException(status_code=503, detail="GPU worker is not ready")
    job = generation.submit(checked_image)
    return JobResponse(job_id=job.job_id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def job_status(job_id: str) -> JobResponse:
    job = generation.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job.job_id,
        status=job.status,  # type: ignore[arg-type]
        error=job.error,
        model_url=f"/jobs/{job.job_id}/download" if job.status == "completed" else None,
    )


@app.get("/jobs/{job_id}/download", dependencies=[Depends(require_api_key)])
def download(job_id: str):
    job = generation.get(job_id)
    if job is None or job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=404, detail="Generated model not found")
    return FileResponse(job.output_path, media_type="model/gltf-binary", filename=f"{job_id}.glb")
