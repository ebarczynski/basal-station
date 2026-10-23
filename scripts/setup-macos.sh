#!/usr/bin/env bash
# basal-station setup for Apple Silicon Macs (M1–M5).
# Installs the basal engine with the MLX + media extras and basal-station itself.
set -euo pipefail

PY=${PYTHON:-python3}

echo "==> Creating virtualenv (.venv)"
$PY -m venv .venv
source .venv/bin/activate

echo "==> Installing basal engine (MLX + media extras)"
BASAL_VERSION=${BASAL_VERSION:-v2.3.5}
uv pip install "basal[mlx,media,mcp] @ https://github.com/rkinas/basal/archive/refs/tags/${BASAL_VERSION}.tar.gz"

echo "==> Installing basal-station"
uv pip install -e .

echo
echo "Done. Next steps:"
echo "  1. Download the Apple Silicon weights (≈13 GB):"
echo "       basal-station download omni-1.0-max --format mlx"
echo "  2. Point the app at your knowledge palace (config/basal-station.example.yaml):"
echo "       cp config/basal-station.example.yaml basal-station.yaml"
echo "       # edit knowledge.command, e.g. ['/Users/you/Documents/knowledge-palace/.venv/bin/kp', 'mcp']"
echo "  3. Launch:"
echo "       basal-station serve     # UI at http://127.0.0.1:8800"
