#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — Run All Baselines
#
# 8 baselines: Naive, Template, UniCoM, EZSwitch, Few-Shot,
#              MCE/MADGF, SwitchLingua 1.0, Persona-Only
#
# Usage:
#   bash baselines/run_all.sh \
#       --api-base http://localhost:8001/v1 \
#       --model Qwen3.5-122B-A10B-FP8 \
#       --lang-pair yue_en \
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
    echo "Usage: bash baselines/run_all.sh --api-base URL --model MODEL [--lang-pair PAIR] [--num N]"
    exit 1
fi

OUTPUT_DIR="output/baselines/${LANG_PAIR}"
mkdir -p "$OUTPUT_DIR"

COMMON="--api-base $API_BASE_VAL --model $MODEL --lang-pair $LANG_PAIR --num-dialogues $NUM"

echo "============================================================"
echo "Running all baselines for $LANG_PAIR ($NUM dialogues each)"
echo "API: $API_BASE_VAL"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

echo ""
echo ">>> [1/8] Naive Prompting"
python baselines/naive_prompting.py $COMMON --output "$OUTPUT_DIR/naive.jsonl"

echo ""
echo ">>> [2/8] Template-Based (Pratapa 2018)"
python baselines/template_based.py $COMMON --output "$OUTPUT_DIR/template.jsonl"

echo ""
echo ">>> [3/8] UniCoM/SWORDS (EMNLP 2025)"
python baselines/unicom_swords.py $COMMON --output "$OUTPUT_DIR/unicom.jsonl"

echo ""
echo ">>> [4/8] EZSwitch (Kuwanto 2024)"
python baselines/ezswitch_style.py $COMMON --output "$OUTPUT_DIR/ezswitch.jsonl"

echo ""
echo ">>> [5/8] Few-Shot Prompting (Yong 2023)"
python baselines/fewshot_prompting.py $COMMON --output "$OUTPUT_DIR/fewshot.jsonl"

echo ""
echo ">>> [6/8] MCE/MADGF (ICASSP 2025)"
python baselines/mce_madgf.py $COMMON --output "$OUTPUT_DIR/mce.jsonl"

echo ""
echo ">>> [7/8] SwitchLingua 1.0 (NeurIPS 2025)"
python baselines/switchlingua1.py $COMMON --output "$OUTPUT_DIR/switchlingua1.jsonl"

echo ""
echo ">>> [8/8] Persona-Only (Ablation)"
python baselines/persona_only.py $COMMON --output "$OUTPUT_DIR/persona_only.jsonl"

echo ""
echo "============================================================"
echo "All baselines complete. Running auto metrics comparison..."
echo "============================================================"

python stage1.5/auto_metrics.py \
    --input "$OUTPUT_DIR/naive.jsonl" \
            "$OUTPUT_DIR/template.jsonl" \
            "$OUTPUT_DIR/unicom.jsonl" \
            "$OUTPUT_DIR/ezswitch.jsonl" \
            "$OUTPUT_DIR/fewshot.jsonl" \
            "$OUTPUT_DIR/mce.jsonl" \
            "$OUTPUT_DIR/switchlingua1.jsonl" \
            "$OUTPUT_DIR/persona_only.jsonl" \
    --labels "Naive" "Template" "UniCoM" "EZSwitch" "Few-Shot" "MCE" "SwitchLingua1.0" "Persona-Only" \
    --lang-pair "$LANG_PAIR" \
    --report "$OUTPUT_DIR/comparison_report.md"

echo ""
echo "Report: $OUTPUT_DIR/comparison_report.md"
echo "Done!"
