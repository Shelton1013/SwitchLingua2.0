#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — SLURM Job: Speech Synthesis (superpod)
# Adapted for HKUST SuperPOD cluster
#
# Usage:
#   sbatch superpod/job_synthesize.slurm
#   sbatch --export=LANG_PAIR=de_en,LIMIT=50 superpod/job_synthesize.slurm
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

# ---- Paths ----
PROJECT_DIR="$HOME/SwitchLingua2.0"
MODEL_DIR="$HOME/models/fish-speech-1.5"
INPUT_FILE="${PROJECT_DIR}/output/${LANG_PAIR}_dialogues.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/output/stage2/${LANG_PAIR}"
ASSET_DIR="${PROJECT_DIR}/superpod/asset"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ---- Environment ----
source ~/.bash_profile
eval "$(conda shell.bash hook)"
conda activate switchlingua

echo "============================================================"
echo "SwitchLingua 2.0 — Speech Synthesis (SuperPOD)"
echo "============================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Lang pair: $LANG_PAIR"
echo "Input:     $INPUT_FILE"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"

# ---- Verify ----
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input not found: $INPUT_FILE"
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found: $MODEL_DIR"
    exit 1
fi

NUM_DIALOGUES=$(wc -l < "$INPUT_FILE")
echo "Input dialogues: $NUM_DIALOGUES"

# ---- Run ----
cd "$PROJECT_DIR"

LIMIT_ARG=""
[ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"

python superpod/synthesize_offline.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --asset-dir "$ASSET_DIR" \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    $LIMIT_ARG

echo "Done at $(date)"
