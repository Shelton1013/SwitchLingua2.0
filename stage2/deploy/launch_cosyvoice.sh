#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — CosyVoice TTS Server Launch Script
#
# Usage:
#   bash launch_cosyvoice.sh [model_dir] [port] [gpu_id] [--vllm]
#
#   # Basic
#   bash launch_cosyvoice.sh /data/models/Fun-CosyVoice3-0.5B-2512 50000 9
#
#   # With vLLM acceleration
#   bash launch_cosyvoice.sh /data/models/Fun-CosyVoice3-0.5B-2512 50000 9 --vllm
#
# Prerequisites:
#   conda activate cosyvoice
#   pip install fastapi uvicorn torchaudio
#   # For vLLM: pip install vllm==0.11.0 transformers==4.57.1
# ============================================================

set -e

MODEL_DIR="${1:-/data/models/Fun-CosyVoice3-0.5B-2512}"
PORT="${2:-50000}"
GPU_ID="${3:-9}"
VLLM_FLAG="${4:-}"

COSYVOICE_ROOT="${COSYVOICE_ROOT:-$HOME/CosyVoice}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -d "$COSYVOICE_ROOT" ]; then
    echo "Error: CosyVoice repo not found: $COSYVOICE_ROOT"
    echo "Clone: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE_ROOT"
    exit 1
fi

echo ">>> CosyVoice TTS Server"
echo ">>> Model: $MODEL_DIR"
echo ">>> Port:  $PORT"
echo ">>> GPU:   $GPU_ID"
echo ">>> vLLM:  ${VLLM_FLAG:-disabled}"

export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_DIR/cosyvoice_server.py" \
    --model_dir "$MODEL_DIR" \
    --port "$PORT" \
    $VLLM_FLAG
