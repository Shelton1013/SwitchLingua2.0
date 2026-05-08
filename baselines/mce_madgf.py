"""
Baseline: MCE/MADGF (Xie & Chen, ICASSP 2025)

Multi-Agent Data Generation Framework — 3 个 Agent 的流水线：
  1. Creator Agent: 选择话题 + 生成具体实例
  2. Engineer Agent: 基于话题和实例生成 CS 文本
  3. Reflector Agent: 反思+修正+过滤

原论文针对粤语-英语 (Cantonese-English)，这里泛化到任意语言对。

Usage:
    python baselines/mce_madgf.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair yue_en \
        --num-dialogues 200 \
        --output output/baselines/yue_en/mce.jsonl
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
logger = logging.getLogger("mce_madgf")

LANG_NAMES = {
    "yue_en": ("Cantonese", "English", "Hong Kong"),
    "zh_en": ("Chinese", "English", "mainland China"),
    "ja_en": ("Japanese", "English", "Japan"),
    "ko_en": ("Korean", "English", "South Korea"),
    "de_en": ("German", "English", "Germany"),
    "fr_en": ("French", "English", "France"),
    "es_en": ("Spanish", "English", "Latin America"),
    "ar_en": ("Arabic", "English", "the Middle East"),
    "hi_en": ("Hindi", "English", "India"),
    "th_en": ("Thai", "English", "Thailand"),
    "ru_en": ("Russian", "English", "Russia"),
    "it_en": ("Italian", "English", "Italy"),
    "ms_en": ("Malay", "English", "Malaysia"),
    "tr_en": ("Turkish", "English", "Turkey"),
}

# 18 topics from the original MCE paper
MCE_TOPICS = [
    "Weather", "Food", "Travel", "Entertainment", "Sports",
    "Local News", "Shopping", "Study", "Work", "Health and Fitness",
    "Pets", "Technology and News", "Movies and TV Shows",
    "Music and Art", "Hobbies and Interests", "History and Literature",
    "Social Media", "Environment",
]


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


def creator_agent(llm, l1_name, l2_name, region, topic):
    """Agent 1: Creator — 生成具体的话题实例（一句示例 CS 文本）"""
    system_prompt = (
        f"You are a creative content planner for bilingual {l1_name}-{l2_name} data."
    )
    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Generate one specific example sentence that a bilingual speaker from {region} "
        f"would say about this topic, naturally mixing {l1_name} and {l2_name}. "
        f"This will serve as a style reference for further generation.\n\n"
        f"Output ONLY the example sentence, nothing else."
    )
    try:
        return llm.chat(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"Creator failed: {e}")
        return None


def engineer_agent(llm, l1_name, l2_name, region, topic, instance):
    """Agent 2: Engineer — 基于话题和实例批量生成 CS 文本"""
    system_prompt = (
        f"As a {region} local, you seamlessly blend {l1_name} with {l2_name} "
        f"in your everyday conversations."
    )
    user_prompt = (
        f"Topic: {topic}\n"
        f"Style reference: {instance}\n\n"
        f"Generate a new code-switching sentence about this topic in the same style. "
        f"It should be natural, like something you would actually say in daily life. "
        f"Mix {l1_name} and {l2_name} naturally.\n\n"
        f"Output ONLY the sentence, nothing else."
    )
    try:
        raw = llm.chat(system_prompt, user_prompt)
        return raw.strip().strip('"').strip("'").strip(""").strip(""")
    except Exception as e:
        logger.error(f"Engineer failed: {e}")
        return None


def reflector_agent(llm, l1_name, l2_name, region, text):
    """Agent 3: Reflector — 反思、修正、评分"""
    system_prompt = (
        f"You are a bilingual language quality reviewer specializing in "
        f"{l1_name}-{l2_name} code-switching from {region}."
    )
    user_prompt = (
        f"Review and improve this code-switching text:\n\n"
        f"\"{text}\"\n\n"
        f"Check for:\n"
        f"1. Grammar correctness in both {l1_name} and {l2_name}\n"
        f"2. Natural code-switching patterns (authentic to {region} speakers)\n"
        f"3. Overall fluency and readability\n\n"
        f"If the text needs improvement, output the improved version.\n"
        f"If it's already good, output it as-is.\n"
        f"Then on a new line, output a score from 1-10.\n\n"
        f"Format:\n"
        f"TEXT: [improved text]\n"
        f"SCORE: [1-10]"
    )
    try:
        raw = llm.chat(system_prompt, user_prompt, temperature=0.3)
        # Parse text and score
        improved_text = text
        score = 5
        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("TEXT:"):
                improved_text = line[5:].strip().strip('"').strip("'")
            elif line.upper().startswith("SCORE:"):
                try:
                    score = int(line[6:].strip().split("/")[0].strip())
                except ValueError:
                    score = 5
        return improved_text, score
    except Exception as e:
        logger.error(f"Reflector failed: {e}")
        return text, 5


def generate_mce(llm, l1_name, l2_name, region, quality_threshold=6):
    """MCE/MADGF full pipeline: Creator → Engineer → Reflector"""

    topic = random.choice(MCE_TOPICS)

    # Step 1: Creator generates instance
    instance = creator_agent(llm, l1_name, l2_name, region, topic)
    if not instance:
        return None, topic

    # Step 2: Engineer generates CS text
    text = engineer_agent(llm, l1_name, l2_name, region, topic, instance)
    if not text:
        return None, topic

    # Step 3: Reflector reviews and improves (2 passes, per paper)
    for pass_num in range(2):
        text, score = reflector_agent(llm, l1_name, l2_name, region, text)
        if score >= quality_threshold:
            break

    # Quality filter
    if score < quality_threshold:
        logger.info(f"  Filtered (score={score} < {quality_threshold})")
        return None, topic

    return text, topic


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: MCE/MADGF (ICASSP 2025) — 3-Agent Pipeline")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="yue_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--quality-threshold", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lang_info = LANG_NAMES.get(args.lang_pair, ("Language1", "English", "bilingual community"))
    l1_name, l2_name, region = lang_info
    llm = LLMClient(args.api_base, args.api_key, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = 0
    filtered = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_dialogues):
            logger.info(f"[{i+1}/{args.num_dialogues}]")

            text, topic = generate_mce(
                llm, l1_name, l2_name, region, args.quality_threshold
            )
            if not text:
                filtered += 1
                continue

            dlg_id = f"MCE_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "mce_madgf",
                "language_pair": args.lang_pair.split("_"),
                "topic": topic,
                "formality": "unknown",
                "relationship": "unknown",
                "speaker_a": {"archetype_id": "none", "persona_description": f"{region} bilingual"},
                "speaker_b": {"archetype_id": "none", "persona_description": "unspecified"},
                "turns": [
                    {"turn": 1, "speaker": "A", "text": text}
                ],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            success += 1

    logger.info(f"Done: {success} generated, {filtered} filtered, saved to {args.output}")


if __name__ == "__main__":
    main()
