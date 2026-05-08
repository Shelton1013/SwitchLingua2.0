#!/bin/bash
# ============================================================
# Stage 2 Offline — Environment Setup
# Run once after first login.
#
# Usage: bash stage2_offline/setup.sh [v2|v1.5]
# ============================================================

set -e
VERSION="${1:-v2}"

echo "=== Stage 2 Offline Setup (Fish Speech $VERSION) ==="

# 1. Create/activate conda env
if ! conda env list | grep -q switchlingua; then
    echo "[1/3] Creating conda environment..."
    conda create -n switchlingua python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate switchlingua

# 2. Install dependencies
echo "[2/3] Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install soundfile numpy

if [ "$VERSION" = "v2" ]; then
    echo "Installing Fish Speech v2 (S2-Pro)..."
    pip install fish-speech
    # If pip install fails:
    # git clone https://github.com/fishaudio/fish-speech.git ~/fish-speech
    # cd ~/fish-speech && pip install -e .
else
    echo "Installing Fish Speech v1.5..."
    git clone --branch v1.5 https://github.com/fishaudio/fish-speech.git ~/fish-speech-1.5
    cd ~/fish-speech-1.5 && pip install -e .
    cd -
fi

# 3. Download model
echo "[3/3] Downloading model..."
pip install huggingface_hub

if [ "$VERSION" = "v2" ]; then
    MODEL_DIR="$HOME/models/s2-pro"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/s2-pro', local_dir='$MODEL_DIR')
print(f'Model downloaded to: $MODEL_DIR')
"
else
    MODEL_DIR="$HOME/models/fish-speech-1.5"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/fish-speech-1.5', local_dir='$MODEL_DIR')
print(f'Model downloaded to: $MODEL_DIR')
"
fi

# Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
import soundfile; print('soundfile: OK')
"

echo ""
echo "=== Setup Complete ==="
echo "Model: $MODEL_DIR"
echo "Version: $VERSION"
echo ""
echo "Next steps:"
echo "  1. Put reference audio in ~/SwitchLingua2.0/stage2_offline/asset/"
echo "  2. Put Stage 1 JSONL in ~/SwitchLingua2.0/output/"
echo "  3. Run: python stage2_offline/synthesize.py --input ... --output ... --asset-dir ... --model-dir $MODEL_DIR --version $VERSION"
echo "  4. Or submit: sbatch stage2_offline/job.slurm"
