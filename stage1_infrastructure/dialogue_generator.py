"""
SwitchLingua 2.0 — 多轮对话 CS 数据生成框架
Multi-Turn Dialogue CS Data Generation Framework

核心设计：
  两个具有不同 CS 画像的 SpeakerAgent 进行多轮对话。
  每个 Agent 有自己的 Persona + Archetype，独立调用 LLM 生成各自的发言。
  对话过程中引入 Giles (1991) Communication Accommodation Theory：
  说话人会根据对方的语言模式动态调整自己的 CS 行为（趋同/保持/趋异）。

RAG 替代方案 — Self-Example Bank（自举示例库）：
  不从质量参差不齐的真实语料中检索，而是：
  1. 首批数据无 few-shot 示例，纯靠 Persona prompt 引导生成
  2. 生成的数据经 rule-based 评估器打分
  3. 高分样本（≥8.0）自动入库，按 archetype × topic 索引
  4. 后续生成时检索同类最佳示例作为 few-shot，持续提升质量
  优势：质量可控、任意语言对可用、与目标风格一致、随规模增长

使用方式：
  python dialogue_generator.py \\
      --num-dialogues 100 \\
      --turns-per-dialogue 6 \\
      --api-base http://localhost:8000/v1 \\
      --model qwen2.5-72b-instruct \\
      --output output/dialogues.jsonl \\
      --seed 42
"""

import json
import random
import time
import hashlib
import logging
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from collections import defaultdict

import yaml

# 本地模块
from sampling import ContextualSampler, SamplingResult
from prompt_generator import (
    CS_BEHAVIOR_TEMPLATE_MAP,
    AGE_DESCRIPTIONS,
    REGION_DESCRIPTIONS,
    PROFICIENCY_DESCRIPTIONS,
    FORMALITY_DESCRIPTIONS,
    L2_AFFINITY_MAP,
    RELATIONSHIP_TO_PERSON,
)
from evaluator_agents import RuleBasedEvaluatorPipeline, TextAnalysis

logger = logging.getLogger("dialogue_generator")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SpeakerConfig:
    """一个对话参与者的完整配置"""
    name: str                       # 显示名（如 "A", "B"）
    sampling_result: SamplingResult # 从 ContextualSampler 采样的完整结果
    accommodation_tendency: float   # 适应倾向：0.0=完全不适应, 1.0=完全趋同
    # 对话过程中的动态状态
    current_cmi_shift: float = 0.0  # 当前 CMI 偏移量（受对方影响）

@dataclass
class DialogueTurn:
    """一轮对话"""
    turn_number: int        # 轮次（从 1 开始）
    speaker_name: str       # 说话人名
    text: str               # 发言内容
    eval_score: float       # rule-based 评估分数
    eval_decision: str      # "pass" / "fail" / "review"
    cmi: float              # 该轮的 CMI
    num_switches: int       # 切换点数量
    metadata: dict = field(default_factory=dict)

@dataclass
class Dialogue:
    """一段完整对话"""
    dialogue_id: str                # 唯一 ID
    turns: list[DialogueTurn]       # 对话轮次列表
    speaker_a: dict                 # Speaker A 的元信息
    speaker_b: dict                 # Speaker B 的元信息
    topic: str                      # 话题
    formality: str                  # 正式度
    relationship: str               # 对话者关系
    language_pair: tuple[str, str]  # 语言对
    avg_score: float = 0.0          # 所有轮次的平均分
    total_turns: int = 0

@dataclass
class DialogueConfig:
    """对话生成的全局配置"""
    num_dialogues: int = 100        # 生成多少段对话
    turns_per_dialogue: int = 6     # 每段对话的轮数
    language_pair: tuple = ("zh", "en")  # 语言对
    # LLM 配置
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"          # vLLM 默认不需要 key
    model: str = "qwen2.5-72b-instruct"
    temperature: float = 0.85
    max_tokens: int = 512
    # 质量控制
    min_turn_score: float = 5.0     # 单轮最低分（低于此分重新生成）
    max_retries: int = 3            # 单轮最大重试次数
    # 适应性
    accommodation_mode: str = "mixed"  # "convergent"/"divergent"/"maintain"/"mixed"
    # Self-Example Bank
    example_bank_path: str = ""     # 自举示例库路径（为空则不使用）
    example_bank_min_score: float = 8.0  # 入库最低分
    num_few_shot: int = 2           # 检索几条示例作为 few-shot
    # 输出
    output_path: str = "output/dialogues.jsonl"
    seed: int = 42


# ============================================================
# LLM 客户端（OpenAI 兼容接口）
# ============================================================

class LLMClient:
    """
    通用 LLM 客户端，支持所有 OpenAI 兼容 API。
    包括：vLLM, TGI, Ollama, DeepSeek, Qwen DashScope, OpenAI 等。

    使用 requests 而非 openai SDK，减少依赖。
    """

    def __init__(self, api_base: str, api_key: str = "EMPTY", model: str = ""):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

        # 延迟导入 requests（服务器上一定有）
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("需要安装 requests: pip install requests")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.85,
        max_tokens: int = 512,
    ) -> str:
        """
        调用 LLM 生成文本。

        参数：
        - system_prompt: 角色设定（包含 persona + CS 行为描述）
        - user_prompt: 生成指令（包含对话历史 + 本轮生成要求）
        - temperature: 采样温度
        - max_tokens: 最大生成长度

        返回：
        - 生成的文本字符串
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            resp = self._requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise


# ============================================================
# Self-Example Bank（自举示例库）
# ============================================================

class SelfExampleBank:
    """
    替代传统 RAG 的自举示例库。

    设计理念：
    - 不从质量参差的真实语料中检索（SEAME 等转录噪声大、仅中英）
    - 而是用自己生成的高分样本作为 few-shot 示例
    - 按 archetype_id × topic_id 索引，确保检索到风格匹配的示例
    - 随生成规模增长，示例库质量和覆盖度持续提升

    存储格式（JSONL）：
    每行一条记录：
    {
        "archetype_id": "ARC_01",
        "topic": "technology",
        "formality": "casual",
        "speaker_text": "...",
        "eval_score": 8.5,
        "cmi": 0.18,
        "dialogue_context": "前一轮对方说了什么（可选）"
    }
    """

    def __init__(self, bank_path: str = "", min_score: float = 8.0):
        self.bank_path = Path(bank_path) if bank_path else None
        self.min_score = min_score
        # 内存索引：{(archetype_id, topic) -> [example, ...]}
        self._index: dict[tuple, list[dict]] = defaultdict(list)
        # 从文件加载已有示例
        if self.bank_path and self.bank_path.exists():
            self._load()

    def _load(self):
        """从 JSONL 文件加载已有示例"""
        count = 0
        with open(self.bank_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (entry.get("archetype_id", ""), entry.get("topic", ""))
                self._index[key].append(entry)
                count += 1
        logger.info(f"Self-Example Bank 加载了 {count} 条示例")

    def add(
        self,
        archetype_id: str,
        topic: str,
        formality: str,
        speaker_text: str,
        eval_score: float,
        cmi: float,
        dialogue_context: str = "",
    ):
        """
        将一条高分样本加入示例库。
        只有 eval_score >= min_score 的样本才会被接受。
        """
        if eval_score < self.min_score:
            return

        entry = {
            "archetype_id": archetype_id,
            "topic": topic,
            "formality": formality,
            "speaker_text": speaker_text,
            "eval_score": eval_score,
            "cmi": round(cmi, 3),
            "dialogue_context": dialogue_context,
        }

        key = (archetype_id, topic)
        self._index[key].append(entry)

        # 追加写入文件
        if self.bank_path:
            self.bank_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bank_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def retrieve(
        self, archetype_id: str, topic: str, k: int = 2
    ) -> list[dict]:
        """
        检索与当前生成条件最匹配的 few-shot 示例。

        检索策略（按优先级）：
        1. 精确匹配：同 archetype + 同 topic
        2. 放宽话题：同 archetype + 任意 topic
        3. 放宽原型：任意 archetype + 同 topic
        4. 无匹配：返回空列表
        """
        # 1. 精确匹配
        candidates = self._index.get((archetype_id, topic), [])
        if len(candidates) >= k:
            # 按分数降序，取 top-k
            sorted_candidates = sorted(
                candidates, key=lambda x: x["eval_score"], reverse=True
            )
            return sorted_candidates[:k]

        # 2. 放宽话题：同 archetype
        all_same_archetype = []
        for (arc, _), examples in self._index.items():
            if arc == archetype_id:
                all_same_archetype.extend(examples)
        if len(all_same_archetype) >= k:
            sorted_candidates = sorted(
                all_same_archetype, key=lambda x: x["eval_score"], reverse=True
            )
            return sorted_candidates[:k]

        # 3. 放宽原型：同 topic
        all_same_topic = []
        for (_, tp), examples in self._index.items():
            if tp == topic:
                all_same_topic.extend(examples)
        if all_same_topic:
            sorted_candidates = sorted(
                all_same_topic, key=lambda x: x["eval_score"], reverse=True
            )
            return sorted_candidates[:k]

        # 4. 返回已有的部分匹配
        return candidates[:k] if candidates else []

    @property
    def total_examples(self) -> int:
        return sum(len(v) for v in self._index.values())


# ============================================================
# Accommodation Controller — Giles (1991) CAT 适应控制器
# ============================================================

class AccommodationController:
    """
    基于 Giles (1991) Communication Accommodation Theory 的适应控制器。

    控制对话过程中说话人 CS 模式的动态变化：
    - convergent (趋同): 听到对方混合较多英文 → 自己也增加英文
    - divergent (趋异): 听到对方混合较多英文 → 自己反而减少英文（标记身份差异）
    - maintain (保持): 不受对方影响，保持自己的模式
    - mixed (混合): 随机分配 convergent/maintain，模拟真实群体的多样性

    实现方式：
    - 跟踪每轮的 CMI 观测值
    - 计算对方的 CMI 趋势
    - 根据适应模式，调整当前说话人的 CMI 偏移量
    - 偏移量体现在 prompt 中的语言混合程度指令里
    """

    # CMI 偏移对应的自然语言指令
    SHIFT_INSTRUCTIONS = {
        "increase": "注意：你的对话伙伴刚才使用了较多英文，你也自然地增加了一些英文的使用。",
        "decrease": "注意：你的对话伙伴刚才使用的英文较少，你也自然地减少了英文的使用。",
        "maintain": "",  # 不加额外指令
        "strong_increase": "注意：你的对话伙伴一直在大量使用英文，你受到影响也明显增加了英文，"
                           "比你平时的习惯更多一些。",
        "strong_decrease": "注意：对方几乎不用英文，你也有意识地减少了英文使用，以配合对方。",
    }

    def __init__(self, mode: str = "mixed"):
        """
        参数：
        - mode: "convergent" / "divergent" / "maintain" / "mixed"
        """
        self.mode = mode
        # 记录每个说话人的历史 CMI
        self._cmi_history: dict[str, list[float]] = defaultdict(list)

    def observe(self, speaker_name: str, cmi: float):
        """记录一个说话人某轮的 CMI"""
        self._cmi_history[speaker_name].append(cmi)

    def get_accommodation_instruction(
        self,
        current_speaker: SpeakerConfig,
        other_speaker_name: str,
    ) -> str:
        """
        根据对方最近的 CS 模式，生成适应性指令。

        返回一段自然语言描述，追加到 prompt 中，引导 LLM 调整 CS 行为。
        """
        other_history = self._cmi_history.get(other_speaker_name, [])
        if not other_history:
            return ""  # 第一轮，无参照

        # 计算对方最近 2 轮的平均 CMI
        recent_other_cmi = sum(other_history[-2:]) / len(other_history[-2:])

        # 计算自己的基准 CMI（原型 CMI 范围的中值）
        cmi_range = current_speaker.sampling_result.archetype.CMI_range
        base_cmi = (cmi_range[0] + cmi_range[1]) / 2

        # 计算差异
        diff = recent_other_cmi - base_cmi

        # 确定适应方向
        tendency = current_speaker.accommodation_tendency
        effective_mode = self.mode
        if effective_mode == "mixed":
            # 根据 speaker 的 accommodation_tendency 决定
            effective_mode = "convergent" if tendency > 0.5 else "maintain"

        if effective_mode == "maintain":
            return self.SHIFT_INSTRUCTIONS["maintain"]

        if effective_mode == "convergent":
            if diff > 0.15:
                return self.SHIFT_INSTRUCTIONS["strong_increase"]
            elif diff > 0.05:
                return self.SHIFT_INSTRUCTIONS["increase"]
            elif diff < -0.15:
                return self.SHIFT_INSTRUCTIONS["strong_decrease"]
            elif diff < -0.05:
                return self.SHIFT_INSTRUCTIONS["decrease"]
            return ""

        if effective_mode == "divergent":
            # 趋异：方向相反
            if diff > 0.10:
                return self.SHIFT_INSTRUCTIONS["decrease"]
            elif diff < -0.10:
                return self.SHIFT_INSTRUCTIONS["increase"]
            return ""

        return ""


# ============================================================
# SpeakerAgent — 对话参与者
# ============================================================

class SpeakerAgent:
    """
    对话中的一个说话人。

    封装了：
    - 说话人的 persona + archetype + language mode
    - System prompt 的构建
    - 对话历史的管理
    - 每轮的 LLM 调用
    """

    def __init__(self, config: SpeakerConfig, language_pair: tuple = ("zh", "en")):
        self.config = config
        self.language_pair = language_pair
        self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建该说话人的 system prompt（角色设定 + CS 行为描述）"""
        result = self.config.sampling_result

        # 角色设定
        persona_desc = result.demographic.persona_description
        proficiency_desc = PROFICIENCY_DESCRIPTIONS.get(
            result.demographic.L2_proficiency, ""
        )
        role_part = f"你是一个{persona_desc}。{proficiency_desc}\n"

        # CS 行为描述
        domain_words = result.situation.domain_words
        topic_label = result.situation.topic_label
        if domain_words:
            words_str = "、".join(domain_words[:5])
            domain_context = (
                f"在谈论{topic_label}相关话题时，会自然地使用英文词汇如 {words_str} 等"
            )
        else:
            domain_context = f"在谈论{topic_label}相关话题时，偶尔会使用一些英文词汇"

        arc_id = result.archetype.id
        cs_template = CS_BEHAVIOR_TEMPLATE_MAP.get(
            arc_id, CS_BEHAVIOR_TEMPLATE_MAP["ARC_01"]
        )
        cs_part = cs_template.format(domain_context=domain_context)

        # 语言混合程度
        level_desc = result.language_mode.description
        cs_level_part = f"\n【语言混合程度】{level_desc}"

        self.system_prompt = (role_part + cs_part + cs_level_part).strip()
        return self.system_prompt

    def build_turn_prompt(
        self,
        dialogue_history: list[dict],
        turn_number: int,
        total_turns: int,
        topic_label: str,
        formality_desc: str,
        relationship_desc: str,
        accommodation_instruction: str = "",
        few_shot_examples: list[dict] = None,
    ) -> str:
        """
        构建某一轮的 user prompt。

        包含：
        1. 对话背景描述
        2. Self-Example Bank 示例（如有）
        3. 对话历史
        4. 适应性指令（如有）
        5. 本轮生成要求

        参数：
        - dialogue_history: 之前所有轮次 [{"speaker": "A", "text": "..."}, ...]
        - turn_number: 当前轮次
        - total_turns: 总轮数
        - topic_label: 话题
        - formality_desc: 正式度描述
        - relationship_desc: 关系描述
        - accommodation_instruction: 适应性指令
        - few_shot_examples: Self-Example Bank 检索到的示例
        """
        parts = []

        # 1. 对话背景
        parts.append(
            f"你正在和一个{relationship_desc}进行关于「{topic_label}」的对话。"
            f"对话氛围{formality_desc}。"
        )

        # 2. Self-Example Bank 示例
        if few_shot_examples:
            parts.append("\n以下是类似场景中自然的语言混合对话示例，供你参考风格（不要照搬内容）：")
            for i, ex in enumerate(few_shot_examples, 1):
                ctx = ex.get("dialogue_context", "")
                ctx_str = f"（对方说：{ctx}）\n" if ctx else ""
                parts.append(f"示例{i}：{ctx_str}{ex['speaker_text']}")
            parts.append("")

        # 3. 对话历史
        if dialogue_history:
            parts.append("以下是目前的对话内容：")
            for turn in dialogue_history:
                parts.append(f"{turn['speaker']}：{turn['text']}")
            parts.append("")

        # 4. 适应性指令
        if accommodation_instruction:
            parts.append(accommodation_instruction)

        # 5. 本轮生成指令
        is_first = turn_number == 1
        is_last = turn_number == total_turns

        if is_first:
            parts.append(
                f"请你作为 {self.config.name} 开始这段对话。"
                f"用你自然的说话方式发起一个话题。"
            )
        elif is_last:
            parts.append(
                f"请你作为 {self.config.name} 回应对方，"
                f"并自然地结束这段对话（比如总结、告别等）。"
            )
        else:
            parts.append(
                f"请你作为 {self.config.name} 自然地回应对方。"
                f"可以回答对方的问题、提出自己的看法、追问、或自然地延伸话题。"
            )

        parts.append(
            "\n要求：\n"
            "- 只输出你这一轮说的话，不要加角色标记、引号或解释\n"
            "- 语言混合要自然流畅，像真实口语\n"
            "- 长度 2-5 句话\n"
            "- 可以包含犹豫词（嗯、那个、like、you know）和自我修正"
        )

        return "\n".join(parts)


# ============================================================
# DialogueGenerator — 主生成器
# ============================================================

class DialogueGenerator:
    """
    多轮对话 CS 数据生成器。

    流程：
    1. 为每段对话采样两个 Speaker 的画像（Persona + Archetype）
    2. 初始化对话上下文（话题、正式度、关系）
    3. 轮流调用 LLM 生成每轮对话
    4. 每轮用 rule-based 评估器打分，不合格则重试
    5. 高分样本加入 Self-Example Bank
    6. 输出结构化 JSONL

    使用方式：
        config = DialogueConfig(
            num_dialogues=100,
            turns_per_dialogue=6,
            api_base="http://localhost:8000/v1",
            model="qwen2.5-72b-instruct",
        )
        generator = DialogueGenerator(config)
        generator.run()
    """

    def __init__(self, config: DialogueConfig):
        self.config = config
        random.seed(config.seed)

        # 初始化各组件
        self.sampler = ContextualSampler()
        self.llm = LLMClient(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model,
        )
        self.evaluator = RuleBasedEvaluatorPipeline()
        self.accommodation = AccommodationController(mode=config.accommodation_mode)

        # Self-Example Bank
        self.example_bank = SelfExampleBank(
            bank_path=config.example_bank_path,
            min_score=config.example_bank_min_score,
        )

        # 统计计数
        self.stats = {
            "dialogues_generated": 0,
            "turns_generated": 0,
            "turns_retried": 0,
            "turns_failed": 0,
            "examples_banked": 0,
        }

    def _sample_speaker_pair(self) -> tuple[SpeakerConfig, SpeakerConfig]:
        """
        采样两个说话人的画像。

        确保两个人共享相同的话题和关系类型，但 Persona 和 Archetype 不同，
        这样对话才有不同的 CS 风格碰撞。
        """
        # 先采样 Speaker A
        result_a = self.sampler.sample()

        # Speaker B 采样：尽量用不同的 persona，但共享话题和关系
        # 尝试最多 10 次找到不同 persona
        result_b = None
        for _ in range(10):
            candidate = self.sampler.sample()
            # 要求不同 persona 或不同 archetype
            if (candidate.demographic.persona_id != result_a.demographic.persona_id
                    or candidate.archetype.id != result_a.archetype.id):
                result_b = candidate
                break
        if result_b is None:
            result_b = self.sampler.sample()  # fallback

        # 确定适应倾向
        mode = self.config.accommodation_mode
        if mode == "convergent":
            tend_a, tend_b = 0.7, 0.7
        elif mode == "divergent":
            tend_a, tend_b = 0.2, 0.2
        elif mode == "maintain":
            tend_a, tend_b = 0.0, 0.0
        else:  # mixed
            tend_a = random.uniform(0.3, 0.8)
            tend_b = random.uniform(0.3, 0.8)

        speaker_a = SpeakerConfig(
            name="A",
            sampling_result=result_a,
            accommodation_tendency=tend_a,
        )
        speaker_b = SpeakerConfig(
            name="B",
            sampling_result=result_b,
            accommodation_tendency=tend_b,
        )

        return speaker_a, speaker_b

    def _extract_speaker_meta(self, config: SpeakerConfig) -> dict:
        """提取说话人的元信息用于输出"""
        r = config.sampling_result
        return {
            "name": config.name,
            "persona_id": r.demographic.persona_id,
            "persona_description": r.demographic.persona_description,
            "archetype_id": r.archetype.id,
            "archetype_name": r.archetype.name,
            "archetype_name_zh": r.archetype.name_zh,
            "region": r.demographic.region_label,
            "age_group": r.demographic.age_group,
            "profession": r.demographic.profession_label,
            "L2_proficiency": r.demographic.L2_proficiency_label,
            "language_mode_level": r.language_mode.level,
            "effective_cmi": round(r.language_mode.effective_cmi, 3),
            "accommodation_tendency": config.accommodation_tendency,
        }

    def _generate_dialogue_id(self, index: int) -> str:
        """生成唯一对话 ID"""
        seed_str = f"{self.config.seed}_{index}_{time.time()}"
        return f"DLG_{hashlib.md5(seed_str.encode()).hexdigest()[:12]}"

    def generate_one_dialogue(self, dialogue_index: int) -> Optional[Dialogue]:
        """
        生成一段完整的多轮对话。

        流程：
        1. 采样两个 speaker
        2. 确定共享上下文（话题、关系、正式度）
        3. 轮流生成每轮对话：
           a. 构建 prompt（含对话历史 + 适应指令 + few-shot 示例）
           b. 调用 LLM 生成
           c. rule-based 评估，不合格则重试
           d. 更新适应控制器
           e. 高分样本入库
        4. 返回结构化 Dialogue 对象
        """
        # 1. 采样 speaker pair
        speaker_a_cfg, speaker_b_cfg = self._sample_speaker_pair()
        agent_a = SpeakerAgent(speaker_a_cfg, self.config.language_pair)
        agent_b = SpeakerAgent(speaker_b_cfg, self.config.language_pair)

        # 2. 共享上下文（取 Speaker A 的话题和关系作为对话上下文）
        ctx = speaker_a_cfg.sampling_result
        topic_label = ctx.situation.topic_label
        topic_id = ctx.situation.topic
        formality = ctx.situation.formality
        formality_desc = FORMALITY_DESCRIPTIONS.get(formality, "轻松随意")
        relationship = ctx.situation.interlocutor_label
        rel_person = RELATIONSHIP_TO_PERSON.get(relationship, relationship)

        dialogue_id = self._generate_dialogue_id(dialogue_index)
        turns: list[DialogueTurn] = []
        history: list[dict] = []  # [{"speaker": "A", "text": "..."}, ...]

        # 重置适应控制器的历史
        self.accommodation._cmi_history.clear()

        # 3. 逐轮生成
        for turn_num in range(1, self.config.turns_per_dialogue + 1):
            # 轮流：奇数轮 A 说，偶数轮 B 说
            if turn_num % 2 == 1:
                current_agent = agent_a
                current_cfg = speaker_a_cfg
                other_name = "B"
            else:
                current_agent = agent_b
                current_cfg = speaker_b_cfg
                other_name = "A"

            # 获取适应性指令
            accommodation_instr = self.accommodation.get_accommodation_instruction(
                current_cfg, other_name
            )

            # 从 Self-Example Bank 检索 few-shot 示例
            few_shot = []
            if self.config.num_few_shot > 0 and self.example_bank.total_examples > 0:
                few_shot = self.example_bank.retrieve(
                    archetype_id=current_cfg.sampling_result.archetype.id,
                    topic=topic_id,
                    k=self.config.num_few_shot,
                )

            # 构建 user prompt
            user_prompt = current_agent.build_turn_prompt(
                dialogue_history=history,
                turn_number=turn_num,
                total_turns=self.config.turns_per_dialogue,
                topic_label=topic_label,
                formality_desc=formality_desc,
                relationship_desc=rel_person,
                accommodation_instruction=accommodation_instr,
                few_shot_examples=few_shot,
            )

            # 带重试的 LLM 生成 + 评估
            best_text = ""
            best_score = 0.0
            best_eval = None

            for retry in range(self.config.max_retries):
                try:
                    generated_text = self.llm.chat(
                        system_prompt=current_agent.system_prompt,
                        user_prompt=user_prompt,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                except Exception as e:
                    logger.warning(f"LLM 调用失败 (retry {retry}): {e}")
                    continue

                # 清理输出（去掉可能的角色标记）
                cleaned = self._clean_output(generated_text, current_cfg.name)

                if not cleaned:
                    continue

                # 规则评估
                eval_result = self.evaluator.evaluate(
                    text=cleaned,
                    archetype_id=current_cfg.sampling_result.archetype.id,
                    archetype_cmi_range=current_cfg.sampling_result.archetype.CMI_range,
                    formality=formality,
                    region=current_cfg.sampling_result.demographic.region,
                    domain_words=current_cfg.sampling_result.situation.domain_words,
                )

                if eval_result.final_score > best_score:
                    best_text = cleaned
                    best_score = eval_result.final_score
                    best_eval = eval_result

                # 达到阈值就不再重试
                if eval_result.final_score >= self.config.min_turn_score:
                    break
                else:
                    self.stats["turns_retried"] += 1

            # 如果所有重试都未通过最低阈值，用最好的一次
            if not best_text or best_eval is None:
                logger.warning(
                    f"对话 {dialogue_id} 第 {turn_num} 轮生成失败，跳过"
                )
                self.stats["turns_failed"] += 1
                continue

            # 更新适应控制器
            self.accommodation.observe(current_cfg.name, best_eval.analysis.cmi)

            # 记录本轮
            turn = DialogueTurn(
                turn_number=turn_num,
                speaker_name=current_cfg.name,
                text=best_text,
                eval_score=best_score,
                eval_decision=best_eval.decision,
                cmi=round(best_eval.analysis.cmi, 3),
                num_switches=len(best_eval.analysis.switch_points),
                metadata={
                    "archetype_id": current_cfg.sampling_result.archetype.id,
                    "retries": retry,
                },
            )
            turns.append(turn)
            history.append({"speaker": current_cfg.name, "text": best_text})
            self.stats["turns_generated"] += 1

            # 高分样本入 Self-Example Bank
            if best_score >= self.config.example_bank_min_score:
                # 对话上下文 = 上一轮对方说的话
                prev_context = ""
                if len(history) >= 2:
                    prev_context = history[-2]["text"][:100]

                self.example_bank.add(
                    archetype_id=current_cfg.sampling_result.archetype.id,
                    topic=topic_id,
                    formality=formality,
                    speaker_text=best_text,
                    eval_score=best_score,
                    cmi=best_eval.analysis.cmi,
                    dialogue_context=prev_context,
                )
                self.stats["examples_banked"] += 1

        # 组装 Dialogue 对象
        if not turns:
            return None

        avg_score = sum(t.eval_score for t in turns) / len(turns)
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            turns=turns,
            speaker_a=self._extract_speaker_meta(speaker_a_cfg),
            speaker_b=self._extract_speaker_meta(speaker_b_cfg),
            topic=topic_label,
            formality=formality,
            relationship=relationship,
            language_pair=self.config.language_pair,
            avg_score=round(avg_score, 2),
            total_turns=len(turns),
        )

        self.stats["dialogues_generated"] += 1
        return dialogue

    @staticmethod
    def _clean_output(text: str, speaker_name: str) -> str:
        """
        清理 LLM 输出：
        - 去掉开头可能的角色标记（"A：", "Speaker A:", 等）
        - 去掉引号包裹
        - 去掉元描述（"以下是...", "好的，" 等）
        """
        text = text.strip()

        # 去掉角色标记
        prefixes = [
            f"{speaker_name}：", f"{speaker_name}:", f"{speaker_name} ：",
            f"{speaker_name} :", f"Speaker {speaker_name}:",
            f"说话人{speaker_name}：",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 去掉常见的 LLM 元描述开头
        meta_prefixes = [
            "好的，", "以下是", "当然，", "没问题，",
            "Sure, ", "Here's ", "Okay, ",
        ]
        for prefix in meta_prefixes:
            if text.startswith(prefix) and len(text) > len(prefix) + 10:
                text = text[len(prefix):].strip()

        # 去掉首尾引号
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith('"') and text.endswith('"')):
            text = text[1:-1].strip()

        return text

    def _dialogue_to_dict(self, dialogue: Dialogue) -> dict:
        """将 Dialogue 对象转为可序列化的字典"""
        return {
            "dialogue_id": dialogue.dialogue_id,
            "topic": dialogue.topic,
            "formality": dialogue.formality,
            "relationship": dialogue.relationship,
            "language_pair": list(dialogue.language_pair),
            "avg_score": dialogue.avg_score,
            "total_turns": dialogue.total_turns,
            "speaker_a": dialogue.speaker_a,
            "speaker_b": dialogue.speaker_b,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "speaker": t.speaker_name,
                    "text": t.text,
                    "eval_score": t.eval_score,
                    "eval_decision": t.eval_decision,
                    "cmi": t.cmi,
                    "num_switches": t.num_switches,
                }
                for t in dialogue.turns
            ],
        }

    def run(self):
        """
        执行批量对话生成。

        输出 JSONL 格式，每行一段完整对话。
        """
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"开始生成 {self.config.num_dialogues} 段对话，"
            f"每段 {self.config.turns_per_dialogue} 轮"
        )
        logger.info(f"模型: {self.config.model}")
        logger.info(f"输出: {output_path}")
        logger.info(
            f"Self-Example Bank: "
            f"{'启用' if self.config.example_bank_path else '未启用'} "
            f"(已有 {self.example_bank.total_examples} 条示例)"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(self.config.num_dialogues):
                try:
                    dialogue = self.generate_one_dialogue(i)
                except Exception as e:
                    logger.error(f"对话 {i} 生成异常: {e}")
                    continue

                if dialogue is None:
                    logger.warning(f"对话 {i} 生成失败（无有效轮次）")
                    continue

                # 写入 JSONL
                line = json.dumps(
                    self._dialogue_to_dict(dialogue), ensure_ascii=False
                )
                f.write(line + "\n")
                f.flush()

                # 进度日志
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        f"进度: {i + 1}/{self.config.num_dialogues} | "
                        f"轮次: {self.stats['turns_generated']} | "
                        f"重试: {self.stats['turns_retried']} | "
                        f"入库: {self.stats['examples_banked']} | "
                        f"当前对话均分: {dialogue.avg_score:.1f}"
                    )

        # 最终统计
        logger.info("=" * 60)
        logger.info("生成完成！最终统计：")
        for k, v in self.stats.items():
            logger.info(f"  {k}: {v}")
        logger.info(
            f"  Self-Example Bank 总量: {self.example_bank.total_examples}"
        )
        logger.info(f"  输出文件: {output_path}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Multi-Turn Dialogue Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基本配置
    parser.add_argument("--num-dialogues", type=int, default=100,
                        help="生成对话数量")
    parser.add_argument("--turns-per-dialogue", type=int, default=6,
                        help="每段对话的轮数")
    parser.add_argument("--language-pair", nargs=2, default=["zh", "en"],
                        help="语言对，如 zh en")

    # LLM 配置
    parser.add_argument("--api-base", type=str,
                        default="http://localhost:8000/v1",
                        help="LLM API 地址（OpenAI 兼容）")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API Key")
    parser.add_argument("--model", type=str,
                        default="qwen2.5-72b-instruct",
                        help="模型名称")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成长度")

    # 质量控制
    parser.add_argument("--min-turn-score", type=float, default=5.0,
                        help="单轮最低评分（低于此分重试）")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单轮最大重试次数")

    # 适应性
    parser.add_argument("--accommodation", type=str, default="mixed",
                        choices=["convergent", "divergent", "maintain", "mixed"],
                        help="Giles CAT 适应模式")

    # Self-Example Bank
    parser.add_argument("--example-bank", type=str, default="",
                        help="Self-Example Bank 文件路径（JSONL）。"
                             "为空则不使用。首次运行会自动创建。")
    parser.add_argument("--example-bank-min-score", type=float, default=8.0,
                        help="入库最低分")
    parser.add_argument("--num-few-shot", type=int, default=2,
                        help="每轮检索几条 few-shot 示例")

    # 输出
    parser.add_argument("--output", type=str,
                        default="output/dialogues.jsonl",
                        help="输出文件路径（JSONL）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 日志
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 构建配置
    config = DialogueConfig(
        num_dialogues=args.num_dialogues,
        turns_per_dialogue=args.turns_per_dialogue,
        language_pair=tuple(args.language_pair),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_turn_score=args.min_turn_score,
        max_retries=args.max_retries,
        accommodation_mode=args.accommodation,
        example_bank_path=args.example_bank,
        example_bank_min_score=args.example_bank_min_score,
        num_few_shot=args.num_few_shot,
        output_path=args.output,
        seed=args.seed,
    )

    # 运行
    generator = DialogueGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
