#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — CosyVoice TTS Server Launch Script
#
# Uses the OFFICIAL CosyVoice FastAPI server (runtime/python/fastapi/server.py)
# with optional vLLM acceleration for higher throughput.
#
# Usage:
#   bash launch_cosyvoice.sh [model_dir] [port] [gpu_id] [--vllm]
#
#   # Basic (no vLLM)
#   bash launch_cosyvoice.sh /data/models/Fun-CosyVoice3-0.5B-2512 50000 9
#
#   # With vLLM acceleration (recommended for large-scale generation)
#   bash launch_cosyvoice.sh /data/models/Fun-CosyVoice3-0.5B-2512 50000 9 --vllm
#
# Prerequisites:
#   conda activate cosyvoice
#   cd ~/CosyVoice && pip install -r requirements.txt
#   # For vLLM acceleration:
#   pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4
# ============================================================

set -e

MODEL_DIR="${1:-/data/models/Fun-CosyVoice3-0.5B-2512}"
PORT="${2:-50000}"
GPU_ID="${3:-9}"
USE_VLLM="${4:-}"

COSYVOICE_ROOT="${COSYVOICE_ROOT:-$HOME/CosyVoice}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Download with:"
    echo "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \\"
    echo "    FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \\"
    echo "    --local-dir $MODEL_DIR"
    exit 1
fi

if [ ! -d "$COSYVOICE_ROOT" ]; then
    echo "Error: CosyVoice repo not found at: $COSYVOICE_ROOT"
    echo "Clone with: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE_ROOT"
    exit 1
fi

SERVER_PY="$COSYVOICE_ROOT/runtime/python/fastapi/server.py"
if [ ! -f "$SERVER_PY" ]; then
    echo "Error: Official server.py not found at: $SERVER_PY"
    exit 1
fi

echo ">>> CosyVoice TTS Server (Official)"
echo ">>> Model:    $MODEL_DIR"
echo ">>> Port:     $PORT"
echo ">>> GPU:      $GPU_ID"
echo ">>> vLLM:     ${USE_VLLM:-disabled}"
echo ">>> Server:   $SERVER_PY"
echo ""

cd "$COSYVOICE_ROOT"
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

# Create a wrapper script that registers vLLM model then imports server
WRAPPER="/tmp/cosyvoice_vllm_wrapper.py"

if [ "$USE_VLLM" = "--vllm" ]; then
    echo ">>> Registering CosyVoice vLLM model..."
    cat > "$WRAPPER" << 'PYEOF'
import sys, os
# Register vLLM model
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# Now run the official server
server_dir = os.path.join(os.environ["COSYVOICE_ROOT"], "runtime", "python", "fastapi")
sys.path.insert(0, server_dir)
import server
PYEOF
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$WRAPPER" --port "$PORT" --model_dir "$MODEL_DIR"
else
    # Launch without vLLM
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$SERVER_PY" \
        --port "$PORT" \
        --model_dir "$MODEL_DIR"
fi
