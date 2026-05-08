"""
Ablation 4: Full System − Archetype

保留 persona、topic injection、evaluator、accommodation，
但去掉 archetype 行为模板——不告诉 LLM "怎么切换"，只告诉它"谁在说话"。

这等价于 persona_only.py 但保留了 topic injection 和 evaluator。

Usage:
    python baselines/ablation_no_archetype.py \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --lang-pair zh_en \
        --num-dialogues 200 \
        --output output/ablation/zh_en/no_archetype.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_generate"))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "stage1_infrastructure"))

from dialogue_generator import DialogueGenerator, GenerationConfig, SpeakerAgent
import argparse


class NoArchetypeSpeakerAgent(SpeakerAgent):
    """Override system prompt to remove archetype behavior template."""

    def _build_system_prompt(self) -> str:
        r = self.result
        lc = self.lang_config
        persona = r.demographic.persona_description

        # Proficiency description
        if lc and lc.proficiency_descriptions:
            prof_map = {k: v.format(l1_name=lc.l1_name, l2_name=lc.l2_name)
                        for k, v in lc.proficiency_descriptions.items()}
        else:
            prof_map = {}
        proficiency = prof_map.get(r.demographic.L2_proficiency, "")

        # Role line (keep persona)
        l1_name = lc.l1_name if lc else "中文"
        l2_name = lc.l2_name if lc else "英文"
        if lc and lc.role_template:
            role = lc.role_template.format(persona=persona, proficiency=proficiency) + "\n"
        else:
            role = f"你是一个{persona}。{proficiency}\n"

        # NO archetype CS behavior template — just a generic instruction
        cs_part = (
            f"\n你是一个{l1_name}和{l2_name}的双语使用者，"
            f"说话时会自然地混合两种语言。\n"
        )

        # Language mode (keep)
        level = r.language_mode.description
        base = f"{role}{cs_part}\n【语言混合程度】{level}"
        base += (
            "\n\n【重要】你只需要输出角色说的话，用 <reply></reply> 标签包裹。"
            "禁止输出分析、思考过程或任何解释。直接输出对话内容。"
        )
        return base.strip()


class NoArchetypeDialogueGenerator(DialogueGenerator):
    """Use NoArchetypeSpeakerAgent instead of default."""

    def _make_agent(self, name, result):
        import random
        mode = self.config.accommodation_mode
        if mode == "convergent":
            tend = 0.7
        elif mode == "divergent":
            tend = 0.2
        elif mode == "maintain":
            tend = 0.0
        else:
            tend = random.uniform(0.3, 0.8)
        return NoArchetypeSpeakerAgent(name, result, tend, lang_config=self.lang_config)


def main():
    parser = argparse.ArgumentParser(description="Ablation: No Archetype")
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

    NoArchetypeDialogueGenerator(config).run()


if __name__ == "__main__":
    main()
