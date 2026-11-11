# TripoSR GPU Worker

Private FastAPI worker for generating GLB meshes with the official
[VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) pipeline.
It is designed for a CUDA GPU such as a Google Colab Tesla T4.

## Important deployment boundary

Render is suitable for the main API, not CUDA inference. Vercel is suitable for
the frontend, not this worker. Colab provides temporary GPU sessions and is not
guaranteed 24/7 hosting. Use a GPU VM or provider later for persistent service.

## Install locally or in Colab

Colab already provides a CUDA-enabled PyTorch build. The setup script deliberately
does not install another PyTorch package:

```bash
git clone https://github.com/Rupesh2228/triposr-gpu-worker.git
cd triposr-gpu-worker
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
./scripts/setup_colab.sh
```

In Colab, enable **Runtime > Change runtime type > T4 GPU** first. Set a strong
`WORKER_API_KEY` in `.env`; never commit `.env`. The script clones the official
TripoSR repository and builds `torchmcubes` against the CUDA-enabled PyTorch
already present. The first model load downloads `stabilityai/TripoSR` from
Hugging Face; subsequent requests reuse the in-memory model.

The dependency pins for Transformers, Tokenizers, NumPy, and Pillow match the
known successful project setup. They should be tested in the active Colab image
before upgrading, because Colab's preinstalled PyTorch/CUDA versions change.

## Run

```bash
export WORKER_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python worker.py
```

The worker binds to `0.0.0.0:8000` for a tunnel or private network. It refuses
generation when CUDA is unavailable and reports the reason through `/health`.

## API

All endpoints require `X-API-Key: $WORKER_API_KEY` or
`Authorization: Bearer $WORKER_API_KEY`.

```bash
curl -H "X-API-Key: $WORKER_API_KEY" http://localhost:8000/health
curl -H "X-API-Key: $WORKER_API_KEY" http://localhost:8000/gpu-status
curl -X POST -H "X-API-Key: $WORKER_API_KEY" \
  -F image=@chair.png http://localhost:8000/generate
curl -H "X-API-Key: $WORKER_API_KEY" http://localhost:8000/jobs/<job-id>
curl -H "X-API-Key: $WORKER_API_KEY" \
  -o model.glb http://localhost:8000/jobs/<job-id>/download
```

`POST /generate` returns `202` with a job ID. Only one inference is allowed at
a time to protect T4 VRAM. Poll `/jobs/{job_id}` until `completed` or `failed`,
then download the GLB. Uploaded inputs are deleted after processing; completed
jobs and GLBs are removed after `JOB_RETENTION_SECONDS`.

## Connecting the Render backend

Set the Render backend's `AI_URL` to the worker's private/reachable base URL and
configure the same secret as the worker's API key in the backend. The backend
must forward the image and API key; do not put the worker key in React or Vercel
client-side code. For Colab, expose the local port only through an authenticated
private tunnel and rotate the key when the session ends.

## Security

- Keep `.env`, model weights, uploads, and generated GLBs out of Git.
- Use a long random API key and rotate it.
- Set `CORS_ORIGINS` only to actual trusted origins; leave it empty for
  server-to-server use.
- Keep the worker off the public internet where possible.
- The in-memory job registry is intentionally single-process. Use a durable
  queue/object store before running multiple worker replicas.
- API errors do not expose local filesystem paths.

## Tests

The tests cover authentication, health/GPU responses, image validation, model
readiness, job validation, and download behavior without requiring CUDA:

```bash
pytest -q
```
