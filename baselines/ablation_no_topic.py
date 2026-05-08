"""
Ablation 2: Full System − Topic Injection

与完整系统相同，但禁用 TopicRouter（不注入任何真实话题信息）。
验证话题信息注入对内容丰富度的贡献。

Usage:
    python baselines/ablation_no_topic.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/ablation/zh_en/no_topic.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_generate"))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "stage1_infrastructure"))

from dialogue_generator import DialogueGenerator, GenerationConfig
import argparse


class NoTopicDialogueGenerator(DialogueGenerator):
    """Override topic fetching to always return empty."""

    def __init__(self, config):
        super().__init__(config)
        # Replace topic_router.fetch to always return []
        self.topic_router.fetch = lambda *a, **kw: []


def main():
    parser = argparse.ArgumentParser(description="Ablation: No Topic Injection")
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

    NoTopicDialogueGenerator(config).run()


if __name__ == "__main__":
    main()
