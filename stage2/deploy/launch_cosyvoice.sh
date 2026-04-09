#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — Stage 2: CosyVoice 3 Deployment Script
#
# Deploys CosyVoice 3 TTS server for speech synthesis.
#
# Prerequisites:
#   1. NVIDIA GPU with >= 6GB VRAM
#   2. Conda environment with Python 3.10
#   3. CosyVoice repo cloned and dependencies installed
#
# Setup (run once):
#   # Clone repo
#   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
#   cd CosyVoice
#
#   # Create conda env
#   conda create -n cosyvoice python=3.10 -y
#   conda activate cosyvoice
#   pip install -r requirements.txt
#   apt-get install sox libsox-dev  # or: conda install -c conda-forge sox
#
#   # Download model
#   python -c "
#   from modelscope import snapshot_download
#   snapshot_download('iic/CosyVoice2-0.5B', local_dir='/data/models/CosyVoice2-0.5B')
#   "
#   # Or for CosyVoice 3:
#   python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
#                     local_dir='/data/models/Fun-CosyVoice3-0.5B-2512')
#   "
#
# Usage:
#   bash launch_cosyvoice.sh [model_dir] [port] [gpu_id]
#
#   # Default: CosyVoice2-0.5B on port 50000, GPU 9
#   bash launch_cosyvoice.sh
#
#   # Custom model and port
#   bash launch_cosyvoice.sh /data/models/Fun-CosyVoice3-0.5B-2512 50001 0
# ============================================================

set -e

MODEL_DIR="${1:-/data/models/CosyVoice2-0.5B}"
PORT="${2:-50000}"
GPU_ID="${3:-9}"

# Validate model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Please download the model first:"
    echo "  python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$MODEL_DIR')\""
    exit 1
fi

echo ">>> CosyVoice TTS Server"
echo ">>> Model: $MODEL_DIR"
echo ">>> Port: $PORT"
echo ">>> GPU: $GPU_ID"
echo ""

# Find the CosyVoice installation directory
COSYVOICE_DIR="${COSYVOICE_DIR:-$(dirname $(python -c 'import cosyvoice; print(cosyvoice.__file__)' 2>/dev/null || echo '/opt/CosyVoice/cosyvoice/__init__.py'))/../}"

# If cosyvoice is not installed as a package, check common locations
if [ ! -f "$COSYVOICE_DIR/server.py" ]; then
    for dir in /opt/CosyVoice ~/CosyVoice /data/CosyVoice; do
        if [ -f "$dir/server.py" ]; then
            COSYVOICE_DIR="$dir"
            break
        fi
    done
fi

if [ ! -f "$COSYVOICE_DIR/server.py" ]; then
    echo "Error: Cannot find CosyVoice server.py"
    echo "Set COSYVOICE_DIR environment variable to the CosyVoice repo root"
    exit 1
fi

echo ">>> CosyVoice dir: $COSYVOICE_DIR"
echo ">>> Starting server..."

cd "$COSYVOICE_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID python server.py \
    --port $PORT \
    --model_dir "$MODEL_DIR"
