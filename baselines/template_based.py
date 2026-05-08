"""
Baseline 2: Template-Based CS Generation (Pratapa et al., ACL 2018 style)

先生成单语句子，然后用规则在预定义位置插入 L2 词汇。
单句级操作，无多轮对话，无 topic 约束。

Usage:
    python baselines/template_based.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --target-cmi 0.15 \
        --output output/baselines/template_zh_en.jsonl
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
logger = logging.getLogger("template_based")

DOMAIN_WORDS = [
    "meeting", "update", "bug", "server", "code", "data", "software",
    "deadline", "project", "email", "schedule", "report", "feedback",
    "weekend", "shopping", "plan", "holiday", "restaurant", "coffee",
    "movie", "music", "game", "gym", "workout", "app", "phone",
    "paper", "research", "exam", "professor", "hotel", "airport",
]

FIXED_EXPRESSIONS = [
    "by the way", "make sense", "no problem", "you know", "I mean",
    "actually", "basically", "honestly", "anyway", "of course",
]

LANG_NAMES = {
    "zh_en": ("Chinese", "English"),
    "yue_en": ("Cantonese", "English"),
    "ja_en": ("Japanese", "English"),
    "ko_en": ("Korean", "English"),
    "de_en": ("German", "English"),
    "fr_en": ("French", "English"),
    "es_en": ("Spanish", "English"),
    "ar_en": ("Arabic", "English"),
    "hi_en": ("Hindi", "English"),
    "th_en": ("Thai", "English"),
    "ru_en": ("Russian", "English"),
    "it_en": ("Italian", "English"),
    "ms_en": ("Malay", "English"),
    "tr_en": ("Turkish", "English"),
}


class LLMClient:
    def __init__(self, api_base, api_key="empty", model="", timeout=120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, system_prompt, user_prompt, temperature=0.8, max_tokens=512):
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


def generate_monolingual_sentence(llm, l1_name):
    """生成一条纯 L1 句子。"""
    system_prompt = f"You are a native {l1_name} speaker."
    user_prompt = (
        f"Write one natural sentence in {l1_name} about daily life or work. "
        f"Just output the sentence, nothing else."
    )
    try:
        raw = llm.chat(system_prompt, user_prompt)
        return raw.strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"Generate failed: {e}")
        return None


def inject_l2_words(text, target_cmi=0.15):
    """在随机位置插入 L2 词汇，按 target CMI 控制数量。"""
    words = text.split()
    if not words:
        return text

    n_total = len(words) if len(words) > 3 else len(text) // 2
    n_insert = max(1, int(n_total * target_cmi))

    pool = DOMAIN_WORDS + random.sample(FIXED_EXPRESSIONS, min(2, len(FIXED_EXPRESSIONS)))
    positions = sorted(random.sample(range(len(words)), min(n_insert, len(words))))

    for pos in positions:
        words[pos] = words[pos] + " " + random.choice(pool)

    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="Baseline: Template-Based CS (Pratapa 2018)")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="zh_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--target-cmi", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    l1_name, _ = LANG_NAMES.get(args.lang_pair, ("Language1", "English"))
    llm = LLMClient(args.api_base, args.api_key, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_dialogues):
            logger.info(f"[{i+1}/{args.num_dialogues}]")

            mono = generate_monolingual_sentence(llm, l1_name)
            if not mono:
                continue

            cs_text = inject_l2_words(mono, args.target_cmi)

            dlg_id = f"TMPL_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "template_based",
                "language_pair": args.lang_pair.split("_"),
                "topic": "unspecified",
                "target_cmi": args.target_cmi,
                "formality": "unknown",
                "relationship": "unknown",
                "speaker_a": {"archetype_id": "none", "persona_description": "unspecified"},
                "speaker_b": {"archetype_id": "none", "persona_description": "unspecified"},
                "turns": [
                    {"turn": 1, "speaker": "A", "text": cs_text}
                ],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            success += 1

    logger.info(f"Done: {success}/{args.num_dialogues} saved to {args.output}")


if __name__ == "__main__":
    main()
