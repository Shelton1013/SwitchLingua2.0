#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — CosyVoice TTS Server Launch Script
#
# Usage:
#   bash launch_cosyvoice.sh [model_dir] [port] [gpu_id]
#
#   # Default: CosyVoice2-0.5B on port 50000, GPU 9
#   bash launch_cosyvoice.sh
#
#   # CosyVoice 3
#   bash launch_cosyvoice.sh /data/models/Fun-CosyVoice3-0.5B-2512 50000 9
#
# Prerequisites:
#   conda activate cosyvoice
#   pip install fastapi uvicorn torchaudio
#   CosyVoice repo cloned at COSYVOICE_ROOT
# ============================================================

set -e

MODEL_DIR="${1:-/data/models/CosyVoice2-0.5B}"
PORT="${2:-50000}"
GPU_ID="${3:-9}"
COSYVOICE_ROOT="${COSYVOICE_ROOT:-$HOME/CosyVoice}"

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo "Download with:"
    echo "  # CosyVoice 2"
    echo "  python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$MODEL_DIR')\""
    echo ""
    echo "  # CosyVoice 3"
    echo "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir /data/models/Fun-CosyVoice3-0.5B-2512"
    exit 1
fi

echo ">>> CosyVoice TTS Server"
echo ">>> Model:  $MODEL_DIR"
echo ">>> Port:   $PORT"
echo ">>> GPU:    $GPU_ID"
echo ">>> CosyVoice root: $COSYVOICE_ROOT"
echo ""

# Set CosyVoice paths
export COSYVOICE_ROOT="$COSYVOICE_ROOT"
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_DIR/cosyvoice_server.py" \
    --model_dir "$MODEL_DIR" \
    --port "$PORT"
