from io import BytesIO
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, generation
from app.services.generation_service import Job


def client() -> TestClient:
    return TestClient(app)


def auth() -> dict[str, str]:
    return {"X-API-Key": "test-secret"}


def valid_png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 4), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_missing_api_key_is_rejected():
    response = client().get("/health")
    assert response.status_code == 401


def test_health_endpoint_reports_gpu_state():
    response = client().get("/health", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert {"status", "cuda_available", "model_loaded", "gpu"} <= body.keys()


def test_gpu_status_endpoint():
    response = client().get("/gpu-status", headers=auth())
    assert response.status_code == 200
    assert "available" in response.json()


def test_invalid_image_is_rejected_before_generation():
    response = client().post(
        "/generate",
        headers=auth(),
        files={"image": ("bad.png", b"not an image", "image/png")},
    )
    assert response.status_code in {400, 503}


def test_generation_requires_ready_model():
    response = client().post(
        "/generate",
        headers=auth(),
        files={"image": ("image.png", valid_png(), "image/png")},
    )
    assert response.status_code == 503


def test_unknown_job_and_download_are_not_found():
    headers = auth()
    assert client().get("/jobs/not-a-job", headers=headers).status_code == 404
    assert client().get("/jobs/not-a-job/download", headers=headers).status_code == 404


def test_completed_job_status_and_download(tmp_path):
    job_id = "00000000-0000-0000-0000-000000000001"
    output = tmp_path / f"{job_id}.glb"
    output.write_bytes(b"glTF")
    now = time.time()
    job = Job(
        job_id=job_id,
        image_path=tmp_path / "input.png",
        status="completed",
        output_path=output,
        created_at=now,
        updated_at=now,
    )
    with generation.lock:
        generation.jobs[job_id] = job

    headers = auth()
    status_response = client().get(f"/jobs/{job_id}", headers=headers)
    download_response = client().get(f"/jobs/{job_id}/download", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["model_url"] == f"/jobs/{job_id}/download"
    assert download_response.status_code == 200
    assert download_response.content == b"glTF"
