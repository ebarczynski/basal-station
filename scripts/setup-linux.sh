#!/usr/bin/env bash
# basal-station setup for Linux with an NVIDIA GPU (works well on small cards, e.g. 3 GB).
# Installs the basal engine (GGUF route via llama.cpp) and basal-station itself.
set -euo pipefail

PY=${PYTHON:-python3}

echo "==> Creating virtualenv (.venv)"
if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    VIRTUAL_ENV="$PWD/.venv"
else
    $PY -m venv .venv
    VIRTUAL_ENV="$PWD/.venv"
fi
source .venv/bin/activate

echo "==> Installing basal engine (GGUF/CPU route; add [fp8] only for >=16 GB Hopper/Blackwell)"
BASAL_VERSION=${BASAL_VERSION:-v2.3.5}
uv pip install --python "$VIRTUAL_ENV/bin/python" "basal[mcp] @ https://github.com/rkinas/basal/archive/refs/tags/${BASAL_VERSION}.tar.gz"

echo "==> Installing basal-station"
uv pip install --python "$VIRTUAL_ENV/bin/python" -e .

echo "==> Checking for llama.cpp (only needed for the engine's 'llamacpp' server mode)"
if command -v llama-server >/dev/null 2>&1 || command -v llama-cli >/dev/null 2>&1; then
    echo "    llama.cpp found."
else
    cat <<'EOF'
    llama-server not found on PATH. That is fine for the default `gguf` mode:
    the engine runs llama.cpp in process on the GGUF file. A standalone
    llama-server build is only needed if you use the engine's separate
    `llamacpp` mode (--llamacpp-url). To build one with CUDA:

      sudo apt install -y build-essential cmake libcurl4-openssl-dev
      git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
      cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
      sudo cp build/bin/llama-* /usr/local/bin/
EOF
fi

echo
echo "Done. Next steps:"
echo "  1. Download GGUF weights — on a 3 GB card start with the 4.5B model:"
echo "       basal-station download 2.0-4.5B --format gguf --quant Q4_K_M"
echo "     (the 12B max model runs too, as a partial offload: ngl is computed from VRAM)"
echo "  2. Configure:"
echo "       cp config/basal-station.example.yaml basal-station.yaml"
echo "       # edit knowledge.command to your knowledge-palace `kp mcp` path"
echo "  3. Launch:"
echo "       basal-station serve     # UI at http://127.0.0.1:8800"
