import os

os.environ.setdefault("WORKER_API_KEY", "test-secret")
os.environ.setdefault("LOAD_MODEL_ON_STARTUP", "false")
os.environ.setdefault("WORKER_STORAGE_DIR", "/tmp/triposr-gpu-worker-tests")

