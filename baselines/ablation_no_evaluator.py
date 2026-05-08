"""
Ablation 3: Full System − Evaluator Retry

与完整系统相同，但禁用评估器的 retry 机制——直接接受 LLM 的第一次输出。
验证 rule-based evaluator 对质量的贡献。

Usage:
    python baselines/ablation_no_evaluator.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/ablation/zh_en/no_evaluator.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_generate"))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "stage1_infrastructure"))

from dialogue_generator import DialogueGenerator, GenerationConfig
import argparse


class NoRetryDialogueGenerator(DialogueGenerator):
    """Override to accept first output without evaluation retry."""

    def __init__(self, config):
        # Force max_retries = 1 so there's no retry
        config.max_retries = 1
        # Set min_turn_score very low so everything passes
        config.min_turn_score = 0.0
        super().__init__(config)


def main():
    parser = argparse.ArgumentParser(description="Ablation: No Evaluator Retry")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="zh_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--turns-per-dialogue", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = GenerationConfig(
        api_bases=[args.api_base],
        api_key=args.api_key,
        model=args.model,
        lang_pair=args.lang_pair,
        num_dialogues=args.num_dialogues,
        turns_per_dialogue=args.turns_per_dialogue,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )

    NoRetryDialogueGenerator(config).run()


if __name__ == "__main__":
    main()
