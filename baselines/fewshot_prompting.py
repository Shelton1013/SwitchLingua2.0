"""
Baseline 5: Few-Shot Prompting (Yong et al., CALCS@EMNLP 2023; Potter & Yuan, EMNLP 2024)

给 LLM 几条真实 CS 示例，让它模仿生成。单句/短段级，无多轮对话，无 topic 约束。
示例是固定的，LLM 的任务是"照着这个风格再造一条"。

Usage:
    python baselines/fewshot_prompting.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/baselines/fewshot_zh_en.jsonl
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
logger = logging.getLogger("fewshot_prompting")

# Few-shot exemplars per language pair (hand-crafted realistic CS)
FEWSHOT_EXAMPLES = {
    "zh_en": [
        "今天那个meeting好长啊，讲了两个小时的budget allocation，我都快睡着了。",
        "你有没有试过那个app？我觉得UI design还不错，就是loading有点慢。",
        "最近在追一部Netflix的剧，plot twist特别多，每集都很intense。",
        "我那个paper的revision快due了，reviewer的comments还没全部address。",
        "刚试了楼下新开的brunch店，menu选择挺多但portion有点小。",
    ],
    "de_en": [
        "Das Meeting heute war so lang, zwei Stunden budget allocation, ich bin fast eingeschlafen.",
        "Hast du das neue AI Tool ausprobiert? Für simple Tasks ganz okay, aber complex Sachen muss man selber machen.",
        "Der neue Project Manager ist echt strict mit den Deadlines, aber fair enough.",
    ],
    "ko_en": [
        "오늘 meeting 진짜 길었어, budget allocation 두 시간이나 했어.",
        "그 app download 했어? UI가 진짜 clean한데 loading이 좀 느려.",
        "최근에 Netflix에서 새로운 drama 보기 시작했는데, plot twist가 미쳤어.",
    ],
    "ar_en": [
        "الmeeting اليوم كان طويل جداً، ساعتين budget allocation، كنت هموت من الملل.",
        "شفت الapp الجديد؟ الUI design حلو بس الloading بطيء شوية.",
    ],
    "es_en": [
        "El meeting de hoy fue super largo, dos horas de budget allocation, casi me duermo.",
        "Has probado esa nueva app? El UI design está cool pero el loading es lento.",
    ],
    "ja_en": [
        "今日のmeeting長かったね、budget allocationで2時間も話してた。",
        "あのappもう試した？UI designはいいけどloadingがちょっと遅い。",
    ],
    "hi_en": [
        "आज का meeting बहुत लंबा था यार, दो घंटे budget allocation पे discuss किया।",
        "वो new app try किया? UI design अच्छा है but loading slow है।",
    ],
}

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
}

# Generic fallback for languages without specific examples
GENERIC_EXAMPLES = [
    "Example 1: [L1 sentence with 'meeting', 'deadline' mixed in]",
    "Example 2: [L1 sentence with 'app', 'design' mixed in]",
    "Example 3: [L1 sentence with 'project', 'feedback' mixed in]",
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


def generate_fewshot(llm, l1_name, l2_name, lang_pair):
    """给 3 条示例，让 LLM 模仿生成 1 条新的 CS 文本。"""

    examples = FEWSHOT_EXAMPLES.get(lang_pair, GENERIC_EXAMPLES)
    selected = random.sample(examples, min(3, len(examples)))

    examples_text = "\n".join(f"- {ex}" for ex in selected)

    system_prompt = (
        f"You generate natural code-switching text between {l1_name} and {l2_name}. "
        f"Study the examples and generate new text in the same style."
    )

    user_prompt = (
        f"Here are examples of natural {l1_name}-{l2_name} code-switching:\n\n"
        f"{examples_text}\n\n"
        f"Now generate ONE new code-switched sentence in the same style. "
        f"It should be different from the examples above. "
        f"Output ONLY the sentence, nothing else."
    )

    try:
        raw = llm.chat(system_prompt, user_prompt)
        raw = raw.strip().strip('"').strip("'").strip("-").strip("•").strip(""").strip(""")
        return raw if raw else None
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Few-Shot Prompting (Yong 2023 / Potter 2024)")
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

            text = generate_fewshot(llm, l1_name, l2_name, args.lang_pair)
            if not text:
                continue

            dlg_id = f"FEWS_{hashlib.md5(f'{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            output = {
                "dialogue_id": dlg_id,
                "method": "fewshot_prompting",
                "language_pair": args.lang_pair.split("_"),
                "topic": "unspecified",
                "formality": "unknown",
                "relationship": "unknown",
                "num_examples_shown": min(len(FEWSHOT_EXAMPLES.get(args.lang_pair, [])), 3),
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
