"""
Baseline: SwitchLingua 1.0 / LinguaMaster (NeurIPS 2025)

4-Agent 协作框架：
  1. Generator Agent: 基于 persona 生成 CS 对话
  2. Fluency Agent: 评估语法流畅度
  3. Naturalness Agent: 评估 CS 自然度
  4. SocioCulture Agent: 评估社会文化适当性

与 SwitchLingua 2.0 的核心区别：
  - 1.0 用 LLM Agent 做评估 → 存在 self-enhancement bias
  - 2.0 用 Rule-based 评估 → 零 bias
  - 1.0 无 archetype 行为模型 → 无行为多样性
  - 1.0 无 accommodation → 无对话者动态适应
  - 1.0 无 topic injection → 内容可能空洞

Usage:
    python baselines/switchlingua1.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair yue_en \
        --num-dialogues 200 \
        --output output/baselines/yue_en/switchlingua1.jsonl
"""

import json
import time
import random
import logging
import argparse
import hashlib
import sys
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("switchlingua1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_infrastructure"))

try:
    from language_config import LanguagePairConfig
except ImportError:
    LanguagePairConfig = None

LANG_NAMES = {
    "yue_en": ("Cantonese", "English", "Hong Kong"),
    "zh_en": ("Chinese", "English", "China/Singapore"),
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

TOPICS = [
    "technology", "work", "daily life", "food", "entertainment",
    "travel", "academic", "sports", "shopping", "emotions",
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


# ============================================================
# Agent 1: Generator
# ============================================================

def generator_agent(llm, l1_name, l2_name, region, persona_desc, topic, turns=4):
    """Generate a multi-turn CS dialogue with persona."""
    system_prompt = (
        f"You are a bilingual {l1_name}-{l2_name} speaker from {region}. "
        f"You are: {persona_desc}.\n"
        f"Generate natural conversations that mix {l1_name} and {l2_name} "
        f"as a real bilingual person would."
    )
    user_prompt = (
        f"Generate a natural code-switching conversation between two speakers (A and B) "
        f"about {topic}.\n\n"
        f"Requirements:\n"
        f"- {turns} turns (A and B alternate)\n"
        f"- Each turn: 1-3 sentences mixing {l1_name} and {l2_name}\n"
        f"- Natural, like real bilingual speakers\n\n"
        f"Format:\nA: ...\nB: ...\nA: ...\nB: ..."
    )
    try:
        raw = llm.chat(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"Generator failed: {e}")
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


# ============================================================
# Agent 2: Fluency Evaluator
# ============================================================

def fluency_agent(llm, l1_name, l2_name, dialogue_text):
    """Evaluate grammar and fluency (LLM-based, 1.0 style)."""
    system_prompt = (
        f"You are a bilingual language quality assessor for {l1_name}-{l2_name} text."
    )
    user_prompt = (
        f"Evaluate the grammatical correctness and fluency of this code-switched dialogue:\n\n"
        f"{dialogue_text}\n\n"
        f"Rate 1-10. Output ONLY a number."
    )
    try:
        raw = llm.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=16)
        return int("".join(c for c in raw if c.isdigit())[:2]) if raw else 5
    except Exception:
        return 5


# ============================================================
# Agent 3: Naturalness Evaluator
# ============================================================

def naturalness_agent(llm, l1_name, l2_name, dialogue_text):
    """Evaluate CS naturalness (LLM-based, 1.0 style)."""
    system_prompt = (
        f"You are a sociolinguistics expert specializing in {l1_name}-{l2_name} code-switching."
    )
    user_prompt = (
        f"Evaluate how natural the code-switching is in this dialogue. "
        f"Does it sound like something a real bilingual speaker would say?\n\n"
        f"{dialogue_text}\n\n"
        f"Rate 1-10. Output ONLY a number."
    )
    try:
        raw = llm.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=16)
        return int("".join(c for c in raw if c.isdigit())[:2]) if raw else 5
    except Exception:
        return 5


# ============================================================
# Agent 4: SocioCulture Evaluator
# ============================================================

def socioculture_agent(llm, l1_name, l2_name, region, dialogue_text):
    """Evaluate social and cultural appropriateness (LLM-based, 1.0 style)."""
    system_prompt = (
        f"You are a cultural anthropologist specializing in bilingual communities in {region}."
    )
    user_prompt = (
        f"Evaluate the social and cultural appropriateness of this {l1_name}-{l2_name} "
        f"code-switching dialogue for a {region} context:\n\n"
        f"{dialogue_text}\n\n"
        f"Rate 1-10. Output ONLY a number."
    )
    try:
        raw = llm.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=16)
        return int("".join(c for c in raw if c.isdigit())[:2]) if raw else 5
    except Exception:
        return 5


# ============================================================
# Full Pipeline
# ============================================================

def format_dialogue(turns_list):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns_list)


def generate_switchlingua1(llm, l1_name, l2_name, region, persona_desc,
                           topic, turns=4, quality_threshold=6, max_retries=3):
    """SwitchLingua 1.0 pipeline: Generate → Evaluate (3 agents) → Retry if low quality"""

    for attempt in range(max_retries):
        # Step 1: Generate
        turns_list = generator_agent(llm, l1_name, l2_name, region, persona_desc, topic, turns)
        if not turns_list:
            continue

        dlg_text = format_dialogue(turns_list)

        # Step 2: Evaluate with 3 LLM agents
        fluency = fluency_agent(llm, l1_name, l2_name, dlg_text)
        naturalness = naturalness_agent(llm, l1_name, l2_name, dlg_text)
        socioculture = socioculture_agent(llm, l1_name, l2_name, region, dlg_text)

        avg_score = (fluency + naturalness + socioculture) / 3.0

        logger.info(
            f"  Attempt {attempt+1}: flu={fluency} nat={naturalness} "
            f"soc={socioculture} avg={avg_score:.1f}"
        )

        if avg_score >= quality_threshold:
            return turns_list, {
                "fluency": fluency,
                "naturalness": naturalness,
                "socioculture": socioculture,
                "avg_score": round(avg_score, 1),
            }

    # Return best attempt even if below threshold
    if turns_list:
        return turns_list, {
            "fluency": fluency,
            "naturalness": naturalness,
            "socioculture": socioculture,
            "avg_score": round(avg_score, 1),
            "below_threshold": True,
        }
    return None, None


# Simple persona descriptions for languages without full config
SIMPLE_PERSONAS = [
    "a university student",
    "a tech worker",
    "an office worker",
    "a young professional",
    "a teacher",
    "a business owner",
    "a researcher",
    "a content creator",
]


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: SwitchLingua 1.0 (NeurIPS 2025) — 4-Agent Framework")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang-pair", default="yue_en")
    parser.add_argument("--num-dialogues", type=int, default=200)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--quality-threshold", type=float, default=6.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lang_info = LANG_NAMES.get(args.lang_pair, ("Language1", "English", "bilingual community"))
    l1_name, l2_name, region = lang_info
    llm = LLMClient(args.api_base, args.api_key, args.model)

    # Try to load personas from lang_config
    personas_descs = list(SIMPLE_PERSONAS)
    if LanguagePairConfig:
        try:
            cfg = LanguagePairConfig.load(args.lang_pair)
            if cfg.personas:
                personas_descs = [p.get("description", p.get("id", "bilingual speaker"))
                                  for p in cfg.personas]
        except Exception:
            pass

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_dialogues):
            topic = random.choice(TOPICS)
            persona_desc = random.choice(personas_descs)
            logger.info(f"[{i+1}/{args.num_dialogues}] topic={topic}, persona={persona_desc[:30]}")

            turns_list, eval_scores = generate_switchlingua1(
                llm, l1_name, l2_name, region, persona_desc,
                topic, args.turns, args.quality_threshold,
            )
            if not turns_list:
                continue

            dlg_id = f"SL1_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "switchlingua_1.0",
                "language_pair": args.lang_pair.split("_"),
                "topic": topic,
                "formality": "unknown",
                "relationship": "unknown",
                "speaker_a": {
                    "archetype_id": "none",
                    "persona_description": persona_desc,
                },
                "speaker_b": {
                    "archetype_id": "none",
                    "persona_description": "unspecified",
                },
                "llm_eval_scores": eval_scores,
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
