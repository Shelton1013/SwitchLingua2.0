"""
SwitchLingua 2.0 — Stage 1→2 桥接: 自然语言 Prompt 生成器
将 ContextualSampler 的采样结果转化为 LLM 可用的自然语言 Prompt。

核心设计原则：
- 用自然语言描述代替数值参数（LLM 对风格描述的理解和遵循远强于数值）
- Prompt 包含角色设定 + 语言行为描述 + 情境描述 + CS 指令
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sampling import ContextualSampler, SamplingResult


# ============================================================
# Prompt 模板
# ============================================================

ROLE_TEMPLATE = """\
你是一个{age_desc}{region_desc}{profession_desc}。{proficiency_desc}
"""

CS_BEHAVIOR_TEMPLATE_MAP = {
    "ARC_01": """\
你的语言习惯：
- 日常交流以中文为主，但会自然地嵌入英文单词和短语
- {domain_context}
- 常用的英文固定表达包括日常的 "by the way"、"make sense"、"no problem" 等
- 切换通常发生在名词或名词短语的位置，你很少切换整句英文
- 连续英文不超过3个词，绝不说完整英文句子
- 句子的语法结构始终是中文的，英文词汇嵌入在中文句法框架中""",

    "ARC_02": """\
你的语言习惯：
- 你会根据话题和语境在中英文之间自然切换
- {domain_context}
- 切换通常发生在句子之间或从句边界，每种语言内部的语法都是完整的
- 你不会在一个短语中间突然换语言，而是在一个完整的思路单元结束后切换
- 有时候引用别人的话或转述某个概念时，会保持原来的语言""",

    "ARC_03": """\
你的语言习惯：
- 中英文融合是你的自然说话方式，你在同一句话里会自由地在中英文之间切换
- {domain_context}
- 你的切换非常频繁，有时一句话里来回切换好几次
- 名词、动词、形容词都可能用英文，不限于术语
- 这不是因为词汇缺口，而是你的日常语言习惯，哪个词先想到就用哪个""",

    "ARC_04": """\
你的语言习惯：
- 你主要用中文交流，但会在特定时刻策略性地切换英文
- {domain_context}
- 你切换英文的目的包括：强调某个观点、引用别人原话、开玩笑、表达正式立场
- 你的切换不是随机的，每次用英文都有明确的交际目的
- 有时你会用英文来显得更专业或更权威，有时用英文来制造幽默效果""",

    "ARC_05": """\
你的语言习惯：
- 你几乎完全用中文说话
- 英文只出现在品牌名、专有名词、或实在找不到中文对应词的情况
- {domain_context}
- 英文单词在你的话语中非常少见，一段话里最多一两个
- 你不会用英文短语或句子，偶尔出现的英文都是单个词""",

    "ARC_06": """\
你的语言习惯：
- 你的CS方式会根据和谁说话而显著变化
- {domain_context}
- 你会自然地向对方的语言习惯靠拢
- 如果对方中英混合很多，你也会增加英文；如果对方主要说中文，你会收敛英文
- 这种调整是自然的社交适应，不是刻意模仿""",
}

SITUATION_TEMPLATE = """\

当前场景：{situation_desc}
对话氛围：{formality_desc}。
"""

CS_LEVEL_INSTRUCTION = """\

【语言混合程度指令】{level_desc}
"""

# ============================================================
# 生成指令模板（user prompt）
# ============================================================

GENERATION_INSTRUCTION_MONOLOGUE = """\
请你以上面描述的角色身份，用你自然的说话方式，{task_desc}。

要求：
- 直接输出说话内容，不要加任何标注、解释或元描述
- 语言混合必须自然流畅，就像真实口语一样
- 长度：{length_desc}
- 可以包含口语中常见的犹豫、填充词（如"嗯"、"那个"、"like"、"you know"）"""

GENERATION_INSTRUCTION_DIALOGUE = """\
请你以上面描述的角色身份参与一段对话。{dialogue_setup}

要求：
- 直接输出对话内容，每轮用 "A:" / "B:" 标记
- 每个角色的语言混合风格应保持一致且自然
- 长度：{length_desc}
- 可以包含口语中常见的犹豫、填充词、自我修正"""

MONOLOGUE_TASKS = [
    "讲一件你最近经历的事情",
    "分享你对这个话题的看法",
    "吐槽一下最近遇到的烦心事",
    "给朋友推荐一个你喜欢的东西",
    "描述你今天的日程安排",
    "回忆一个有趣的经历",
    "解释一个你最近学到的概念",
    "聊聊你最近在忙什么",
]

DIALOGUE_SETUPS = [
    "两个人正在闲聊近况。",
    "两个人在讨论一个共同话题。",
    "一个人在向另一个人请教问题。",
    "两个人在分享各自的经历和看法。",
    "一个人在给另一个人讲述自己遇到的事情。",
]


# ============================================================
# Prompt 生成器
# ============================================================

# 辅助映射
AGE_DESCRIPTIONS = {
    "18-25": "在读大学或刚毕业的年轻人",
    "25-35": "工作了几年的年轻职场人",
    "35-50": "有丰富工作经验的中年人",
    "50+": "年纪稍大、经验丰富的人",
}

REGION_DESCRIPTIONS = {
    "singapore": "在新加坡生活",
    "malaysia": "在马来西亚生活的华人",
    "hong_kong": "在香港生活",
    "mainland_urban": "在中国大陆一线城市生活",
    "taiwan": "在台湾生活",
    "overseas_diaspora": "在海外（欧美澳）生活的华人",
}

PROFICIENCY_DESCRIPTIONS = {
    "basic": "你的英语水平有限，只掌握一些基本的常见英文单词。",
    "intermediate": "你的英语水平中等，能使用英文短语和简单句子，有一定的词汇量。",
    "advanced": "你的英语很流利，能自如地使用英文表达复杂的想法。",
    "near_native": "你的中英文能力接近均衡，两种语言都很流利自如。",
}

FORMALITY_DESCRIPTIONS = {
    "casual": "场合比较轻松随意，像朋友聊天一样",
    "semi_formal": "场合半正式，像同事交流或课堂讨论",
    "formal": "场合比较正式，需要注意措辞",
}

L2_AFFINITY_MAP = {
    "high": "这个话题涉及很多英文相关的内容",
    "medium-high": "这个话题有不少英文相关的概念和词汇",
    "medium": "这个话题中英文内容都有涉及",
    "medium-low": "这个话题偏中文环境，英文内容较少",
    "low": "这个话题几乎都是中文语境",
}


def _build_domain_context(result: SamplingResult) -> str:
    """根据话题和职业构建领域上下文描述"""
    domain_words = result.situation.domain_words
    topic_label = result.situation.topic_label

    if domain_words:
        words_sample = domain_words[:5]
        words_str = "、".join(words_sample)
        return f"在谈论{topic_label}相关话题时，会自然地使用英文词汇如 {words_str} 等"
    else:
        return f"在谈论{topic_label}相关话题时，偶尔会使用一些英文词汇"


RELATIONSHIP_TO_PERSON = {
    "亲密朋友": "好朋友",
    "同事": "同事",
    "同学": "同学",
    "师生": "老师",
    "陌生人": "刚认识的人",
    "家人": "家人",
}


def _build_situation_desc(result: SamplingResult) -> str:
    """构建情境描述"""
    topic = result.situation.topic_label
    rel = result.situation.interlocutor_label
    person = RELATIONSHIP_TO_PERSON.get(rel, rel)
    dtype = result.situation.dialogue_type

    if dtype == "monologue":
        return f"你正在独自讲述/分享关于{topic}的内容"
    elif dtype == "dyadic":
        return f"你正在和一个{person}讨论{topic}的话题"
    else:
        return f"你正在和几个{person}一起聊{topic}的话题"


def generate_prompt(result: SamplingResult) -> str:
    """
    将采样结果转化为完整的自然语言 Prompt。

    Args:
        result: ContextualSampler.sample() 的输出

    Returns:
        可直接用于 LLM 的自然语言角色说明 + CS 行为描述 + 情境描述
    """
    # 角色设定：优先用 persona_description，更自然
    persona_desc = result.demographic.persona_description
    proficiency_desc = PROFICIENCY_DESCRIPTIONS.get(
        result.demographic.L2_proficiency, ""
    )
    role_part = f"你是一个{persona_desc}。{proficiency_desc}\n"

    # CS 行为描述
    domain_context = _build_domain_context(result)
    arc_id = result.archetype.id
    cs_template = CS_BEHAVIOR_TEMPLATE_MAP.get(arc_id, CS_BEHAVIOR_TEMPLATE_MAP["ARC_01"])
    cs_part = cs_template.format(domain_context=domain_context)

    # 情境描述
    situation_desc = _build_situation_desc(result)
    formality_desc = FORMALITY_DESCRIPTIONS.get(result.situation.formality, "")
    situation_part = SITUATION_TEMPLATE.format(
        situation_desc=situation_desc,
        formality_desc=formality_desc,
    )

    # CS 程度指令
    level_desc = result.language_mode.description
    cs_level_part = CS_LEVEL_INSTRUCTION.format(level_desc=level_desc)

    # 组合
    full_prompt = role_part + cs_part + situation_part + cs_level_part

    return full_prompt.strip()


def generate_user_prompt(result: SamplingResult) -> str:
    """
    生成用于触发 LLM 生成 CS 文本的 user prompt（生成指令）。

    与 generate_prompt() 输出的 system prompt 配合使用：
    - system prompt: 角色设定 + CS 行为描述 + 情境
    - user prompt: 具体的生成任务指令
    """
    dtype = result.situation.dialogue_type

    # 根据 language mode 决定长度
    level = result.language_mode.level
    if level in ("minimal", "light"):
        length_desc = "3-5句话"
    elif level in ("moderate",):
        length_desc = "4-7句话"
    else:
        length_desc = "5-8句话"

    if dtype == "monologue":
        task = random.choice(MONOLOGUE_TASKS)
        return GENERATION_INSTRUCTION_MONOLOGUE.format(
            task_desc=task,
            length_desc=length_desc,
        )
    else:
        setup = random.choice(DIALOGUE_SETUPS)
        return GENERATION_INSTRUCTION_DIALOGUE.format(
            dialogue_setup=setup,
            length_desc=f"共4-6轮对话，每轮{length_desc}",
        )


def generate_full_prompt(result: SamplingResult) -> dict:
    """
    生成完整的 LLM 调用所需的 prompt 组合。

    返回结构：
    {
        "system_prompt": "角色设定 + CS 行为描述 + 情境 + 混合指令",
        "user_prompt": "具体生成任务指令",
        "metadata": { ... }
    }
    """
    system_prompt = generate_prompt(result)
    user_prompt = generate_user_prompt(result)

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "metadata": {
            "persona_id": result.demographic.persona_id,
            "persona_description": result.demographic.persona_description,
            "archetype_id": result.archetype.id,
            "archetype_name": result.archetype.name,
            "archetype_name_zh": result.archetype.name_zh,
            "region": result.demographic.region_label,
            "age_group": result.demographic.age_group,
            "profession": result.demographic.profession_label,
            "L2_proficiency": result.demographic.L2_proficiency_label,
            "topic": result.situation.topic_label,
            "formality": result.situation.formality_label,
            "dialogue_type": result.situation.dialogue_type,
            "relationship": result.situation.interlocutor_label,
            "language_mode_level": result.language_mode.level,
            "effective_cmi": round(result.language_mode.effective_cmi, 3),
        },
    }


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Prompt Generator — 生成自然语言CS角色Prompt"
    )
    parser.add_argument("-n", type=int, default=3, help="Number of prompts to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    sampler = ContextualSampler()
    results = sampler.sample_batch(args.n, seed=args.seed)

    outputs = []
    for i, result in enumerate(results):
        entry = generate_full_prompt(result)
        entry["index"] = i
        outputs.append(entry)

        if not args.output:
            print(f"\n{'='*60}")
            print(f"Sample {i} | {result.archetype.name_zh} ({result.archetype.name})")
            print(f"Language Mode: {result.language_mode.level} "
                  f"(CMI ≈ {result.language_mode.effective_cmi:.3f})")
            print(f"{'='*60}")
            print("\n[SYSTEM PROMPT]")
            print(entry["system_prompt"])
            print("\n[USER PROMPT]")
            print(entry["user_prompt"])

    if args.output:
        output_json = json.dumps(outputs, ensure_ascii=False, indent=2)
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"\nSaved {args.n} prompts to {args.output}")
