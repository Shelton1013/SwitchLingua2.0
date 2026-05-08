#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — Run Ablation Studies
#
# 4 ablation configs + full system as reference
#
# Usage:
#   bash baselines/run_ablation.sh \
#       --api-base http://localhost:8001/v1 \
#       --model Qwen3.5-122B-A10B-FP8 \
#       --lang-pair zh_en \
#       --num 200
# ============================================================

# Parse args
API_BASE_VAL=""
MODEL=""
LANG_PAIR="yue_en"
NUM=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-base) API_BASE_VAL="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --lang-pair) LANG_PAIR="$2"; shift 2;;
        --num) NUM="$2"; shift 2;;
        *) shift;;
    esac
done

if [ -z "$API_BASE_VAL" ] || [ -z "$MODEL" ]; then
    echo "Usage: bash baselines/run_ablation.sh --api-base URL --model MODEL [--lang-pair PAIR] [--num N]"
    exit 1
fi

OUTPUT_DIR="output/ablation/${LANG_PAIR}"
mkdir -p "$OUTPUT_DIR"

COMMON="--api-base $API_BASE_VAL --model $MODEL --lang-pair $LANG_PAIR --num-dialogues $NUM"

echo "============================================================"
echo "Running ablation studies for $LANG_PAIR ($NUM dialogues each)"
echo "API: $API_BASE_VAL"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# --- Full System (reference) ---
echo ""
echo ">>> [1/5] Full System (reference)"
python stage1_generate/dialogue_generator.py \
    $COMMON \
    --turns-per-dialogue 4 \
    --max-tokens 2048 \
    --output "$OUTPUT_DIR/full_system.jsonl"

# --- Ablation 1: No Accommodation ---
echo ""
echo ">>> [2/5] Ablation: − Accommodation"
python baselines/ablation_no_accommodation.py \
    $COMMON \
    --output "$OUTPUT_DIR/no_accommodation.jsonl"

# --- Ablation 2: No Topic Injection ---
echo ""
echo ">>> [3/5] Ablation: − Topic Injection"
python baselines/ablation_no_topic.py \
    $COMMON \
    --output "$OUTPUT_DIR/no_topic.jsonl"

# --- Ablation 3: No Evaluator Retry ---
echo ""
echo ">>> [4/5] Ablation: − Evaluator Retry"
python baselines/ablation_no_evaluator.py \
    $COMMON \
    --output "$OUTPUT_DIR/no_evaluator.jsonl"

# --- Ablation 4: No Archetype ---
echo ""
echo ">>> [5/5] Ablation: − Archetype"
python baselines/ablation_no_archetype.py \
    $COMMON \
    --output "$OUTPUT_DIR/no_archetype.jsonl"

# --- Auto Metrics Comparison ---
echo ""
echo "============================================================"
echo "Running auto metrics comparison..."
echo "============================================================"

python stage1.5/auto_metrics.py \
    --input "$OUTPUT_DIR/full_system.jsonl" \
            "$OUTPUT_DIR/no_accommodation.jsonl" \
            "$OUTPUT_DIR/no_topic.jsonl" \
            "$OUTPUT_DIR/no_evaluator.jsonl" \
            "$OUTPUT_DIR/no_archetype.jsonl" \
    --labels "Full System" "−Accommodation" "−Topic Injection" "−Evaluator" "−Archetype" \
    --lang-pair "$LANG_PAIR" \
    --report "$OUTPUT_DIR/ablation_report.md"

echo ""
echo "Report: $OUTPUT_DIR/ablation_report.md"
echo "Done!"
