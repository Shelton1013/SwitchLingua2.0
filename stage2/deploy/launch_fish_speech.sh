#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — Fish Speech S2 TTS Server
#
# Usage:
#   bash launch_fish_speech.sh [port] [gpu_id]
#
# Setup (run once):
#   conda create -n fish-speech python=3.12 -y
#   conda activate fish-speech
#   git clone https://github.com/fishaudio/fish-speech.git ~/fish-speech
#   cd ~/fish-speech && pip install -e .[cu121]
#   pip install ormsgpack
#   huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
# ============================================================

set -e

PORT="${1:-8080}"
GPU_ID="${2:-9}"
FISH_ROOT="${FISH_ROOT:-$HOME/fish-speech}"
CHECKPOINT="${FISH_CHECKPOINT:-$FISH_ROOT/checkpoints/s2-pro}"

if [ ! -d "$FISH_ROOT" ]; then
    echo "Error: Fish Speech not found at $FISH_ROOT"
    echo "Clone: git clone https://github.com/fishaudio/fish-speech.git $FISH_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Model not found at $CHECKPOINT"
    echo "Download: cd $FISH_ROOT && huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro"
    exit 1
fi

echo ">>> Fish Speech S2 TTS Server"
echo ">>> Model: $CHECKPOINT"
echo ">>> Port:  $PORT"
echo ">>> GPU:   $GPU_ID"

cd "$FISH_ROOT"

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/api_server.py \
    --llama-checkpoint-path "$CHECKPOINT" \
    --decoder-checkpoint-path "$CHECKPOINT/codec.pth" \
    --listen "0.0.0.0:$PORT" \
    --compile
