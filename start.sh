#!/usr/bin/env bash
set -euo pipefail

echo "=== start.sh: setup vLLM server ==="

# ---------------------------
# Create venv
# ---------------------------
if [ ! -d "venv" ]; then
  echo "[1] Creating venv..."
  python3 -m venv venv --without-pip

  echo "[2] Installing pip..."
  curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  ./venv/bin/python get-pip.py
  rm get-pip.py
fi

# ---------------------------
# Activate
# ---------------------------
echo "[3] Activating venv..."
source venv/bin/activate

# ---------------------------
# Upgrade tools
# ---------------------------
echo "[4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ---------------------------
# Install PyTorch (CUDA 12.4)
# ---------------------------
echo "[5] Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

# ---------------------------
# Install vLLM + FastAPI
# ---------------------------
echo "[6] Installing vLLM + dependencies..."
pip install -r requirements.txt

# ---------------------------
# (Optional) HuggingFace auth
# ---------------------------
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------
# Start server
# ---------------------------
echo "=== Starting GPT-OSS vLLM server ==="
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1