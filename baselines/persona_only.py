"""
Baseline 4: Persona-Only Generation (No Archetype)

给 LLM 提供完整的 persona 描述（地区、职业、年龄、L2 水平），但不提供
archetype 行为模板。这是 ablation study 的关键 baseline——验证 archetype
模板的增量贡献。

与 SwitchLingua 2.0 的区别：
- 有 persona → 知道"谁在说话"
- 无 archetype → 不知道"怎么切换"
- 无 accommodation → 无对话者适应
- 有 topic → 知道"聊什么"

Usage:
    python baselines/persona_only.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/baselines/persona_only_zh_en.jsonl
"""

import json
import sys
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
logger = logging.getLogger("persona_only")

# Add infrastructure path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_infrastructure"))

from language_config import LanguagePairConfig


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


def sample_persona(lang_config):
    """Sample a persona from the lang config."""
    personas = lang_config.personas
    if not personas:
        return None
    weights = [p.get("weight", 1.0 / len(personas)) for p in personas]
    return random.choices(personas, weights=weights, k=1)[0]


def generate_persona_only_dialogue(llm, persona_a, persona_b, topic_label,
                                   l1_name, l2_name, turns=4):
    """Generate dialogue with persona info but NO archetype behavioral template."""

    desc_a = persona_a.get("description", "a bilingual speaker")
    desc_b = persona_b.get("description", "a bilingual speaker")
    prof_a = persona_a.get("L2_proficiency", {}).get("label", "intermediate")
    prof_b = persona_b.get("L2_proficiency", {}).get("label", "intermediate")

    system_prompt = (
        f"You generate realistic bilingual code-switching dialogues between "
        f"{l1_name} and {l2_name}.\n\n"
        f"Speaker A: {desc_a}. {l2_name} proficiency: {prof_a}.\n"
        f"Speaker B: {desc_b}. {l2_name} proficiency: {prof_b}.\n\n"
        f"Generate dialogue that naturally mixes {l1_name} and {l2_name}, "
        f"reflecting each speaker's background and proficiency level."
    )

    user_prompt = (
        f"Topic: {topic_label}\n\n"
        f"Generate a {turns}-turn conversation (A and B alternate).\n"
        f"Each turn: 1-3 sentences mixing {l1_name} and {l2_name}.\n\n"
        f"Format:\nA: ...\nB: ...\nA: ...\nB: ..."
    )

    try:
        raw = llm.chat(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None

    turns_list = []
    current_speaker = None
    current_text = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("A:") or line.startswith("A："):
            if current_speaker and current_text:
                turns_list.append({"speaker": current_speaker, "text": " ".join(current_text)})
            current_speaker = "A"
            current_text = [line[2:].strip()]
        elif line.startswith("B:") or line.startswith("B："):
            if current_speaker and current_text:
                turns_list.append({"speaker": current_speaker, "text": " ".join(current_text)})
            current_speaker = "B"
            current_text = [line[2:].strip()]
        elif current_speaker:
            current_text.append(line)
    if current_speaker and current_text:
        turns_list.append({"speaker": current_speaker, "text": " ".join(current_text)})

    return turns_list if len(turns_list) >= 2 else None


def main():
    parser = argparse.ArgumentParser(description="Baseline: Persona-Only (No Archetype)")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="zh_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lang_config = LanguagePairConfig.load(args.lang_pair)
    l1_name = lang_config.l1_name_en
    l2_name = lang_config.l2_name_en

    # Get topic labels
    raw = lang_config.personas_raw or {}
    topics = raw.get("situation_pools", {}).get("topics", [])
    topic_labels = [t["label"] for t in topics] if topics else [
        "technology", "work", "daily life", "food", "entertainment"
    ]

    llm = LLMClient(args.api_base, args.api_key, args.model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_dialogues):
            persona_a = sample_persona(lang_config)
            persona_b = sample_persona(lang_config)
            # Ensure different personas
            for _ in range(10):
                if persona_b["id"] != persona_a["id"]:
                    break
                persona_b = sample_persona(lang_config)

            topic_label = random.choice(topic_labels)
            logger.info(
                f"[{i+1}/{args.num_dialogues}] A={persona_a['id']}, "
                f"B={persona_b['id']}, topic={topic_label}"
            )

            turns_list = generate_persona_only_dialogue(
                llm, persona_a, persona_b, topic_label,
                l1_name, l2_name, args.turns
            )
            if not turns_list:
                continue

            dlg_id = f"PERS_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "persona_only",
                "language_pair": args.lang_pair.split("_"),
                "topic": topic_label,
                "formality": "unknown",
                "relationship": "unknown",
                "speaker_a": {
                    "archetype_id": "none",
                    "persona_id": persona_a["id"],
                    "persona_description": persona_a.get("description", ""),
                    "region": persona_a.get("region", {}).get("label", ""),
                    "L2_proficiency": persona_a.get("L2_proficiency", {}).get("label", ""),
                },
                "speaker_b": {
                    "archetype_id": "none",
                    "persona_id": persona_b["id"],
                    "persona_description": persona_b.get("description", ""),
                    "region": persona_b.get("region", {}).get("label", ""),
                    "L2_proficiency": persona_b.get("L2_proficiency", {}).get("label", ""),
                },
                "turns": [
                    {"turn": j+1, "speaker": t["speaker"], "text": t["text"]}
                    for j, t in enumerate(turns_list)
                ],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            success += 1

    logger.info(f"Done: {success}/{args.num_dialogues} saved to {args.output}")


if __name__ == "__main__":
    main()
