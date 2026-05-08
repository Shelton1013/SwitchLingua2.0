#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — SLURM Job: Offline Speech Synthesis
# Adapted for HKUST SuperPOD cluster
#
# Usage:
#   sbatch stage2_offline/job.slurm
#   sbatch --export=LANG_PAIR=de_en,LIMIT=50 stage2_offline/job.slurm
# ============================================================

#SBATCH -J sl2-tts
#SBATCH -o logs/tts_%j.out
#SBATCH -e logs/tts_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -p project
#SBATCH --account=itscspod

# ---- Configuration (override via --export) ----
LANG_PAIR="${LANG_PAIR:-zh_en}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-cuda:0}"
FS_VERSION="${FS_VERSION:-v2}"

# ---- Paths ----
PROJECT_DIR="$HOME/SwitchLingua2.0"
MODEL_DIR="$HOME/models/s2-pro"          # v2: s2-pro, v1.5: fish-speech-1.5
INPUT_FILE="${PROJECT_DIR}/output/${LANG_PAIR}_dialogues.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/output/stage2/${LANG_PAIR}"
ASSET_DIR="${PROJECT_DIR}/stage2_offline/asset"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ---- Environment ----
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate switchlingua

echo "============================================================"
echo "Stage 2 Offline Synthesis — HKUST SuperPOD"
echo "============================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Lang pair: $LANG_PAIR"
echo "Version:   $FS_VERSION"
echo "Input:     $INPUT_FILE"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"

# ---- Verify ----
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input not found: $INPUT_FILE"
    echo "Available JSONL files:"
    ls "$PROJECT_DIR/output/"*.jsonl 2>/dev/null || echo "  (none)"
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found: $MODEL_DIR"
    exit 1
fi
if [ ! -d "$ASSET_DIR" ] || [ -z "$(ls $ASSET_DIR/*.wav 2>/dev/null)" ]; then
    echo "ERROR: No reference audio in $ASSET_DIR"
    exit 1
fi

# ---- Run ----
cd "$PROJECT_DIR"

LIMIT_ARG=""
[ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"

python stage2_offline/synthesize.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --asset-dir "$ASSET_DIR" \
    --model-dir "$MODEL_DIR" \
    --version "$FS_VERSION" \
    --device "$DEVICE" \
    $LIMIT_ARG

echo "Done at $(date)"
