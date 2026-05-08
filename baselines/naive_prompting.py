"""
Baseline 1: Naive Prompting

最简单的方法——直接告诉 LLM "生成一段 code-switching 文本"。
不提供 topic、不做多轮对话、不提供任何约束。
这是最常见的最低基准线。

Usage:
    python baselines/naive_prompting.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/baselines/naive_zh_en.jsonl
"""

import json
import time
import random
import logging
import argparse
import hashlib
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("naive_prompting")

LANG_NAMES = {
    "zh_en": ("Chinese", "English"),
    "yue_en": ("Cantonese", "English"),
    "ja_en": ("Japanese", "English"),
    "ko_en": ("Korean", "English"),
    "hi_en": ("Hindi", "English"),
    "ar_en": ("Arabic", "English"),
    "th_en": ("Thai", "English"),
    "ru_en": ("Russian", "English"),
    "de_en": ("German", "English"),
    "fr_en": ("French", "English"),
    "es_en": ("Spanish", "English"),
    "it_en": ("Italian", "English"),
    "ms_en": ("Malay", "English"),
    "tr_en": ("Turkish", "English"),
    "min_en": ("Hokkien", "English"),
}


class LLMClient:
    def __init__(self, api_base, api_key="empty", model="", timeout=120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, system_prompt, user_prompt, temperature=0.8, max_tokens=1024):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "qwen" in self.model.lower() or "Qwen" in self.model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def generate_naive(llm, l1_name, l2_name):
    """单次生成一段 CS 文本，无 topic、无多轮、无任何约束。"""

    system_prompt = "You are a helpful assistant."

    user_prompt = (
        f"Generate a short code-switching text that mixes {l1_name} and {l2_name} "
        f"naturally, like a real bilingual person would speak. "
        f"Just output the text directly, nothing else."
    )

    try:
        raw = llm.chat(system_prompt, user_prompt)
        # 清理：去掉可能的引号包裹
        raw = raw.strip().strip('"').strip("'").strip(""").strip(""")
        return raw if raw else None
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Baseline: Naive Prompting")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="zh_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    l1_name, l2_name = LANG_NAMES.get(args.lang_pair, ("Language1", "English"))
    llm = LLMClient(args.api_base, args.api_key, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_dialogues):
            logger.info(f"[{i+1}/{args.num_dialogues}]")

            text = generate_naive(llm, l1_name, l2_name)
            if not text:
                continue

            dlg_id = f"NAIVE_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "naive_prompting",
                "language_pair": args.lang_pair.split("_"),
                "topic": "unspecified",
                "formality": "unknown",
                "relationship": "unknown",
                "speaker_a": {"archetype_id": "none", "persona_description": "unspecified"},
                "speaker_b": {"archetype_id": "none", "persona_description": "unspecified"},
                "turns": [
                    {"turn": 1, "speaker": "A", "text": text}
                ],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            success += 1

    logger.info(f"Done: {success}/{args.num_dialogues} saved to {args.output}")


if __name__ == "__main__":
    main()
