#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — Quick Submit Helper
#
# Usage:
#   bash superpod/submit.sh zh_en          # synthesize zh_en, all dialogues
#   bash superpod/submit.sh de_en 100      # synthesize de_en, first 100
#   bash superpod/submit.sh all            # submit jobs for all lang pairs
# ============================================================

PROJECT_DIR="$HOME/SwitchLingua2.0"
LANG_PAIR="${1:-zh_en}"
LIMIT="${2:-0}"

if [ "$LANG_PAIR" = "all" ]; then
    echo "Submitting jobs for all available language pairs..."
    for f in "$PROJECT_DIR"/output/*_dialogues.jsonl; do
        pair=$(basename "$f" | sed 's/_dialogues.jsonl//')
        echo "  Submitting: $pair"
        sbatch --export=LANG_PAIR=$pair,LIMIT=$LIMIT "$PROJECT_DIR/superpod/job_synthesize.slurm"
    done
    echo "All jobs submitted. Check with: squeue -u $USER"
else
    INPUT="$PROJECT_DIR/output/${LANG_PAIR}_dialogues.jsonl"
    if [ ! -f "$INPUT" ]; then
        echo "ERROR: $INPUT not found"
        echo "Available:"
        ls "$PROJECT_DIR"/output/*_dialogues.jsonl 2>/dev/null | while read f; do
            echo "  $(basename $f .jsonl | sed 's/_dialogues//')"
        done
        exit 1
    fi

    NUM=$(wc -l < "$INPUT")
    echo "Submitting: $LANG_PAIR ($NUM dialogues, limit=$LIMIT)"
    sbatch --export=LANG_PAIR=$LANG_PAIR,LIMIT=$LIMIT "$PROJECT_DIR/superpod/job_synthesize.slurm"
    echo "Submitted. Check with: squeue -u $USER"
fi
