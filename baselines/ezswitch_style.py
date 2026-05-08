"""
Baseline 4: EZSwitch-Style (Kuwanto et al., 2024)

用 Poplack 等价约束告知 LLM 合法的切换位置。单句级生成，无多轮对话，无 topic。
prompt 中只有语法约束，没有 persona 或行为模板。

Usage:
    python baselines/ezswitch_style.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/baselines/ezswitch_zh_en.jsonl
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
logger = logging.getLogger("ezswitch_style")

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

MIXING_LEVELS = [
    {"name": "light", "desc": "about 5-15% English words"},
    {"name": "moderate", "desc": "about 15-30% English words"},
    {"name": "heavy", "desc": "about 30-50% English words"},
]


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


def generate_ezswitch(llm, l1_name, l2_name, mixing):
    """单句级 CS 生成，带等价约束指令。"""

    system_prompt = (
        f"You generate linguistically valid code-switched text between "
        f"{l1_name} and {l2_name}.\n\n"
        f"LINGUISTIC CONSTRAINTS (Equivalence Constraint, Poplack 1980):\n"
        f"1. Switches must occur where both languages have compatible word order.\n"
        f"2. Valid positions: between verb and object, at clause boundaries, "
        f"before/after noun phrases, at conjunction points.\n"
        f"3. INVALID: inside a word, inside a fixed expression, "
        f"between a determiner and its noun in the same language.\n"
        f"4. {l1_name} provides the grammatical frame. "
        f"{l2_name} provides content words."
    )

    user_prompt = (
        f"Generate one code-switched sentence mixing {l1_name} and {l2_name}.\n"
        f"Mixing level: {mixing['desc']}.\n"
        f"Follow the Equivalence Constraint strictly.\n"
        f"Output ONLY the sentence, nothing else."
    )

    try:
        raw = llm.chat(system_prompt, user_prompt)
        raw = raw.strip().strip('"').strip("'").strip(""").strip(""")
        return raw if raw else None
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: EZSwitch-Style (Kuwanto 2024)")
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
            mixing = random.choice(MIXING_LEVELS)
            logger.info(f"[{i+1}/{args.num_dialogues}] mixing={mixing['name']}")

            text = generate_ezswitch(llm, l1_name, l2_name, mixing)
            if not text:
                continue

            dlg_id = f"EZSW_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "ezswitch_style",
                "language_pair": args.lang_pair.split("_"),
                "topic": "unspecified",
                "mixing_level": mixing["name"],
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
