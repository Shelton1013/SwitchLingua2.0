"""
Baseline 3: UniCoM/SWORDS (Sangmin Lee et al., EMNLP 2025 Findings)

POS 标注 + 词级翻译替换。单句级操作，无多轮对话，无 topic 约束。
模拟 SWORDS 算法：对单语句子做词性分析，在名词/动词/形容词位置替换为 L2。

Usage:
    python baselines/unicom_swords.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --target-cmi 0.15 \
        --output output/baselines/unicom_zh_en.jsonl
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
logger = logging.getLogger("unicom_swords")

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

    def chat(self, system_prompt, user_prompt, temperature=0.3, max_tokens=1024):
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


def generate_and_substitute(llm, l1_name, l2_name, target_cmi=0.15):
    """
    SWORDS pipeline:
    1. Generate a monolingual L1 sentence
    2. POS-tag + identify substitutable words
    3. Replace selected words with L2 translations
    """

    # Step 1: Generate monolingual sentence
    mono_prompt = (
        f"Write one natural sentence in {l1_name} about daily life or work. "
        f"Just output the sentence, nothing else."
    )
    try:
        mono = llm.chat(f"You are a native {l1_name} speaker.", mono_prompt, temperature=0.8)
        mono = mono.strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"Generate mono failed: {e}")
        return None

    if not mono:
        return None

    # Step 2+3: POS-tag and substitute (single LLM call)
    word_count = len(mono.split()) if " " in mono else len(mono) // 2
    n_sub = max(1, int(word_count * target_cmi))

    sub_prompt = (
        f"Given this {l1_name} sentence:\n\n\"{mono}\"\n\n"
        f"1. Identify the NOUNS, VERBS, and ADJECTIVES\n"
        f"2. Pick exactly {n_sub} of them (nouns first, then verbs, then adjectives)\n"
        f"3. Replace each picked word with its {l2_name} translation\n"
        f"4. Keep everything else in {l1_name}\n\n"
        f"Output ONLY the final mixed sentence, nothing else."
    )
    try:
        result = llm.chat(
            "You are a bilingual linguist performing word-level substitution.",
            sub_prompt, temperature=0.3
        )
        result = result.strip().strip('"').strip("'").strip(""").strip(""")
        return result if result else mono
    except Exception as e:
        logger.warning(f"Substitute failed: {e}")
        return mono


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: UniCoM/SWORDS (EMNLP 2025) — POS Word Substitution")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="zh_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--target-cmi", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    l1_name, l2_name = LANG_NAMES.get(args.lang_pair, ("Language1", "English"))
    llm = LLMClient(args.api_base, args.api_key, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_dialogues):
            logger.info(f"[{i+1}/{args.num_dialogues}]")

            cs_text = generate_and_substitute(llm, l1_name, l2_name, args.target_cmi)
            if not cs_text:
                continue

            dlg_id = f"UNIC_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "unicom_swords",
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
