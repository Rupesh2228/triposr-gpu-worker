#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ ! -d third_party/TripoSR ]]; then
  git clone https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR
fi

python -m pip uninstall -y torchmcubes >/dev/null 2>&1 || true
python -m pip install --no-build-isolation \
  "git+https://github.com/tatsy/torchmcubes.git"

mkdir -p runtime
echo "Setup complete. Set WORKER_API_KEY and run: python worker.py"

