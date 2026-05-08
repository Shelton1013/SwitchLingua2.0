#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — One-time Environment Setup on Superpod
# Run this manually after first login (NOT as a SLURM job)
#
# Usage: bash superpod/setup_env.sh
# ============================================================

set -e

echo "=== SwitchLingua 2.0 Superpod Environment Setup ==="

# 1. Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n switchlingua python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate switchlingua

# 2. Install Fish Speech
echo "[2/4] Installing Fish Speech..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fish-speech soundfile numpy

# If pip install fails, try from source:
# git clone https://github.com/fishaudio/fish-speech.git ~/fish-speech
# cd ~/fish-speech && pip install -e .

# 3. Download model
echo "[3/4] Downloading Fish Speech 1.5 model..."
MODEL_DIR="$HOME/models/fish-speech-1.5"
mkdir -p "$MODEL_DIR"

# Option A: huggingface-cli (if available)
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir "$MODEL_DIR"
else
    # Option B: pip install huggingface_hub first
    pip install huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/fish-speech-1.5', local_dir='$MODEL_DIR')
"
fi

# 4. Verify
echo "[4/4] Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')

import soundfile
print(f'soundfile: OK')

try:
    import fish_speech
    print(f'fish-speech: OK')
except:
    print('fish-speech: NOT FOUND (try installing from source)')
"

echo ""
echo "=== Setup complete ==="
echo "Model directory: $MODEL_DIR"
echo "Conda environment: switchlingua"
echo ""
echo "Next steps:"
echo "  1. Put reference audio files in ~/SwitchLingua2.0/superpod/asset/"
echo "  2. Put Stage 1 JSONL files in ~/SwitchLingua2.0/output/"
echo "  3. Submit job: sbatch superpod/job_synthesize.slurm"
