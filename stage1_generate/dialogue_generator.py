"""
SwitchLingua 2.0 — 多轮对话 CS 数据生成器（完整版）
Multi-Turn Dialogue CS Data Generator with MCP Information Injection

与 SwitchLingua 1.0 的对比：
  1.0: 给定 topic → 单轮 LLM 生成 CS 文本（无具体信息注入）
  2.0: 给定 topic → MCP 获取真实信息 → 两个 Agent 多轮对话讨论
       + Self-Example Bank 风格参考
       + Giles CAT 适应性动态
       + 每轮 rule-based 质量控制

三层信息注入架构：
  Layer 1 — Topic Information (MCP)：注入真实的、具体的话题信息（新闻/论文/知识）
            让 agent 有"谈资"，对话有实质内容，而非空洞的模板化聊天
  Layer 2 — Self-Example Bank：注入同类高分生成样本作为 CS 风格参考
            解决"怎么说"而非"说什么"的问题，随生成规模提升
  Layer 3 — Accommodation Dynamics：基于 Giles (1991) CAT 理论
            动态调整说话人的 CS 行为，模拟真实多语者的互相适应

本地部署（10× A6000 48GB）：
  # 1. 启动 vLLM 服务
  bash deploy/launch_vllm.sh a   # Qwen3-235B-A22B (推荐)

  # 2. 运行对话生成
  python dialogue_generator.py \\
      --num-dialogues 1000 \\
      --turns-per-dialogue 6 \\
      --model Qwen/Qwen3-235B-A22B \\
      --example-bank data/self_examples.jsonl \\
      --output output/dialogues.jsonl

  # 多实例负载均衡（方案 C: 5× Qwen3-32B）
  python dialogue_generator.py \\
      --api-base http://localhost:8000/v1 http://localhost:8001/v1 \\
                 http://localhost:8002/v1 http://localhost:8003/v1 \\
                 http://localhost:8004/v1 \\
      --model Qwen/Qwen3-32B-Instruct \\
      --num-dialogues 5000
"""

import sys
import json
import random
import time
import hashlib
import logging
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 将 stage1_infrastructure 加入 Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1_infrastructure"))

from sampling import ContextualSampler, SamplingResult
from prompt_generator import (
    CS_BEHAVIOR_TEMPLATE_MAP,
    PROFICIENCY_DESCRIPTIONS,
    FORMALITY_DESCRIPTIONS,
    RELATIONSHIP_TO_PERSON,
)
from evaluator_agents import RuleBasedEvaluatorPipeline

from topic_information import TopicRouter, InformationSnippet

logger = logging.getLogger("dialogue_generator")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DialogueTurn:
    """一轮对话"""
    turn_number: int
    speaker_name: str
    text: str
    eval_score: float
    eval_decision: str
    cmi: float
    num_switches: int
    metadata: dict = field(default_factory=dict)

@dataclass
class DialogueOutput:
    """一段完整对话的输出"""
    dialogue_id: str
    turns: list[DialogueTurn]
    speaker_a: dict
    speaker_b: dict
    topic: str
    topic_id: str
    formality: str
    relationship: str
    language_pair: list
    avg_score: float
    total_turns: int
    injected_info: list[dict]   # 注入的话题信息

@dataclass
class GenerationConfig:
    """生成器配置"""
    # 对话参数
    num_dialogues: int = 100
    turns_per_dialogue: int = 6
    language_pair: tuple = ("zh", "en")
    # LLM（本地 vLLM 部署）
    api_bases: list = field(default_factory=lambda: ["http://localhost:8000/v1"])
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3-235B-A22B"  # 推荐 MoE 模型
    disable_thinking: bool = True          # 关闭 Qwen3 thinking mode
    temperature: float = 0.85
    max_tokens: int = 512
    # 质量控制
    min_turn_score: float = 5.0
    max_retries: int = 3
    # 适应性
    accommodation_mode: str = "mixed"
    # Self-Example Bank
    example_bank_path: str = ""
    example_bank_min_score: float = 8.0
    num_few_shot: int = 2
    # Topic Information
    provider_config_path: str = ""
    max_info_snippets: int = 3
    # 输出
    output_path: str = "output/dialogues.jsonl"
    seed: int = 42


# ============================================================
# LLM Client (OpenAI 兼容)
# ============================================================

class LLMClient:
    """
    本地 vLLM 客户端，支持多端点轮询负载均衡。

    部署方式：
    - 单实例: api_bases=["http://localhost:8000/v1"]
    - 多实例: api_bases=["http://localhost:8000/v1", "http://localhost:8001/v1", ...]
      多实例时自动轮询分发请求，提升吞吐量。

    Qwen3 特殊说明：
    - Qwen3 默认启用 thinking mode（生成 <think>...</think> 标签）
    - 对话生成不需要思考过程，通过 extra_body 中传 chat_template_kwargs
      或在 system prompt 末尾加 /no_think 指令来关闭
    """

    def __init__(self, api_bases: list[str], api_key: str = "EMPTY",
                 model: str = "", disable_thinking: bool = True):
        """
        参数：
        - api_bases: vLLM 端点列表（支持多实例负载均衡）
        - api_key: vLLM 默认不需要 key，填 "EMPTY"
        - model: 模型名（需与 vLLM 启动时一致）
        - disable_thinking: 关闭 Qwen3 的 thinking mode（推荐）
        """
        self.api_bases = [b.rstrip("/") for b in api_bases]
        self.api_key = api_key
        self.model = model
        self.disable_thinking = disable_thinking
        self._call_count = 0  # 用于轮询计数
        import requests
        self._session = requests.Session()

    def _next_endpoint(self) -> str:
        """轮询选择下一个端点"""
        endpoint = self.api_bases[self._call_count % len(self.api_bases)]
        self._call_count += 1
        return endpoint

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.85, max_tokens: int = 512) -> str:
        endpoint = self._next_endpoint()

        # Qwen3 thinking mode 控制
        # 方式 1: 通过 extra_body 参数（vLLM 支持）
        extra_body = {}
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload["extra_body"] = extra_body

        resp = self._session.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=180,  # 本地推理可能较慢，给足超时
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # 安全清理：如果 thinking mode 未成功关闭，去掉 <think> 标签
        if "<think>" in content:
            import re
            content = re.sub(r'<think>.*?</think>', '', content,
                             flags=re.DOTALL).strip()

        return content


# ============================================================
# Self-Example Bank
# ============================================================

class SelfExampleBank:
    """
    自举示例库：用自己生成的高分样本作为 few-shot 风格参考。
    按 archetype × topic 索引，检索 CS 风格一致的示例。
    """

    def __init__(self, path: str = "", min_score: float = 8.0):
        self.path = Path(path) if path else None
        self.min_score = min_score
        self._index: dict[tuple, list[dict]] = defaultdict(list)
        if self.path and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = (entry.get("archetype_id", ""),
                               entry.get("topic", ""))
                        self._index[key].append(entry)

    def add(self, archetype_id: str, topic: str, text: str,
            score: float, cmi: float, context: str = ""):
        if score < self.min_score:
            return
        entry = {"archetype_id": archetype_id, "topic": topic,
                 "speaker_text": text, "eval_score": score,
                 "cmi": round(cmi, 3), "dialogue_context": context}
        self._index[(archetype_id, topic)].append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def retrieve(self, archetype_id: str, topic: str, k: int = 2) -> list[dict]:
        """按 archetype+topic 精确匹配 → 放宽 archetype → 放宽 topic"""
        # 精确匹配
        candidates = self._index.get((archetype_id, topic), [])
        if len(candidates) >= k:
            return sorted(candidates, key=lambda x: x["eval_score"],
                          reverse=True)[:k]
        # 放宽话题
        for (arc, _), exs in self._index.items():
            if arc == archetype_id:
                candidates.extend(exs)
        if len(candidates) >= k:
            return sorted(candidates, key=lambda x: x["eval_score"],
                          reverse=True)[:k]
        return candidates[:k]

    @property
    def total(self) -> int:
        return sum(len(v) for v in self._index.values())


# ============================================================
# Accommodation Controller (Giles 1991 CAT)
# ============================================================

class AccommodationController:
    """
    基于 Communication Accommodation Theory 的 CS 动态适应控制器。
    跟踪对方的 CMI 趋势，生成自然语言适应指令注入 prompt。
    """

    INSTRUCTIONS = {
        "strong_up": "注意：对方一直在大量使用英文，你受到影响也明显增加了英文使用。",
        "up": "注意：对方刚才使用了较多英文，你也自然地增加了一些英文。",
        "down": "注意：对方刚才使用的英文较少，你也自然地减少了英文。",
        "strong_down": "注意：对方几乎不用英文，你也有意识地减少了英文使用。",
    }

    def __init__(self, mode: str = "mixed"):
        self.mode = mode
        self._history: dict[str, list[float]] = defaultdict(list)

    def observe(self, speaker: str, cmi: float):
        self._history[speaker].append(cmi)

    def get_instruction(self, speaker_cmi_range: list, other_name: str,
                        tendency: float) -> str:
        other_hist = self._history.get(other_name, [])
        if not other_hist:
            return ""
        recent = sum(other_hist[-2:]) / len(other_hist[-2:])
        base = (speaker_cmi_range[0] + speaker_cmi_range[1]) / 2
        diff = recent - base

        eff_mode = self.mode
        if eff_mode == "mixed":
            eff_mode = "convergent" if tendency > 0.5 else "maintain"
        if eff_mode == "maintain":
            return ""
        if eff_mode == "convergent":
            if diff > 0.15:
                return self.INSTRUCTIONS["strong_up"]
            elif diff > 0.05:
                return self.INSTRUCTIONS["up"]
            elif diff < -0.15:
                return self.INSTRUCTIONS["strong_down"]
            elif diff < -0.05:
                return self.INSTRUCTIONS["down"]
        return ""

    def reset(self):
        self._history.clear()


# ============================================================
# Speaker Agent
# ============================================================

class SpeakerAgent:
    """
    对话中的一个说话人。
    封装 persona/archetype → system prompt，
    以及每轮 user prompt 的构建。
    """

    def __init__(self, name: str, sampling_result: SamplingResult,
                 accommodation_tendency: float = 0.5):
        self.name = name
        self.result = sampling_result
        self.tendency = accommodation_tendency
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        r = self.result
        persona = r.demographic.persona_description
        proficiency = PROFICIENCY_DESCRIPTIONS.get(r.demographic.L2_proficiency, "")
        role = f"你是一个{persona}。{proficiency}\n"

        # CS 行为描述
        domain_words = r.situation.domain_words
        topic = r.situation.topic_label
        if domain_words:
            domain_ctx = f"在谈论{topic}相关话题时，会自然地使用英文词汇如{'、'.join(domain_words[:5])}等"
        else:
            domain_ctx = f"在谈论{topic}相关话题时，偶尔会使用一些英文词汇"

        cs_template = CS_BEHAVIOR_TEMPLATE_MAP.get(
            r.archetype.id, CS_BEHAVIOR_TEMPLATE_MAP["ARC_01"]
        )
        cs_part = cs_template.format(domain_context=domain_ctx)
        level = r.language_mode.description

        return f"{role}{cs_part}\n\n【语言混合程度】{level}".strip()

    def build_turn_prompt(
        self,
        history: list[dict],
        turn_num: int,
        total_turns: int,
        topic_label: str,
        formality_desc: str,
        relationship_desc: str,
        topic_info_text: str = "",
        accommodation_text: str = "",
        few_shot_examples: list[dict] = None,
    ) -> str:
        """
        构建一轮的 user prompt。

        包含 3 层信息注入：
        1. topic_info_text: MCP 获取的真实话题信息
        2. few_shot_examples: Self-Example Bank 的风格参考
        3. accommodation_text: CAT 适应性指令
        + 对话历史 + 生成指令
        """
        parts = []

        # 对话背景
        parts.append(
            f"你正在和一个{relationship_desc}进行关于「{topic_label}」的对话。"
            f"对话氛围{formality_desc}。"
        )

        # --- Layer 1: MCP 话题信息注入 ---
        if topic_info_text:
            parts.append("")
            parts.append(topic_info_text)

        # --- Layer 2: Self-Example Bank 风格参考 ---
        if few_shot_examples:
            parts.append("")
            parts.append("以下是类似场景中自然的语言混合对话示例，供你参考风格（不要照搬内容）：")
            for i, ex in enumerate(few_shot_examples, 1):
                ctx = ex.get("dialogue_context", "")
                if ctx:
                    parts.append(f"示例{i}（对方说：{ctx[:50]}）：{ex['speaker_text']}")
                else:
                    parts.append(f"示例{i}：{ex['speaker_text']}")

        # 对话历史
        if history:
            parts.append("")
            parts.append("以下是目前的对话内容：")
            for h in history:
                parts.append(f"{h['speaker']}：{h['text']}")

        # --- Layer 3: CAT 适应性指令 ---
        if accommodation_text:
            parts.append("")
            parts.append(accommodation_text)

        # 生成指令
        parts.append("")
        if turn_num == 1:
            parts.append(
                f"请你作为 {self.name} 开始这段对话。"
                f"用你自然的说话方式发起话题。"
            )
        elif turn_num == total_turns:
            parts.append(
                f"请你作为 {self.name} 回应对方，并自然地结束对话。"
            )
        else:
            parts.append(
                f"请你作为 {self.name} 自然地回应对方。"
                f"可以回答、追问、表达看法或延伸话题。"
            )

        parts.append(
            "\n要求：\n"
            "- 只输出你这一轮说的话，不要加角色标记或解释\n"
            "- 语言混合要自然流畅，像真实口语\n"
            "- 长度 2-5 句话\n"
            "- 可以包含犹豫词（嗯、那个、like、you know）和自我修正"
        )

        return "\n".join(parts)


# ============================================================
# Main Generator
# ============================================================

class DialogueGenerator:
    """
    完整的多轮对话 CS 数据生成器。

    生成流程（每段对话）：
    1. 采样两个 Speaker 画像（不同 Persona + Archetype）
    2. 确定共享上下文（话题、正式度、关系）
    3. MCP 获取话题信息（调用 TopicRouter）
    4. 逐轮生成：
       a. 构建 prompt = system(persona) + user(信息+历史+适应+风格)
       b. 调用 LLM
       c. Rule-based 评估，不合格重试
       d. 更新适应状态
       e. 高分样本入 Self-Example Bank
    5. 输出 JSONL
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        random.seed(config.seed)

        self.sampler = ContextualSampler()
        self.llm = LLMClient(
            config.api_bases, config.api_key, config.model,
            config.disable_thinking,
        )
        self.evaluator = RuleBasedEvaluatorPipeline()
        self.accommodation = AccommodationController(config.accommodation_mode)
        self.example_bank = SelfExampleBank(
            config.example_bank_path, config.example_bank_min_score
        )

        # Topic Information Router
        provider_cfg = config.provider_config_path or None
        self.topic_router = TopicRouter(provider_cfg)

        self.stats = Counter()

    def _sample_pair(self) -> tuple[SamplingResult, SamplingResult]:
        """采样两个不同画像的 Speaker"""
        a = self.sampler.sample()
        for _ in range(10):
            b = self.sampler.sample()
            if (b.demographic.persona_id != a.demographic.persona_id
                    or b.archetype.id != a.archetype.id):
                return a, b
        return a, self.sampler.sample()

    def _make_agent(self, name: str, result: SamplingResult) -> SpeakerAgent:
        mode = self.config.accommodation_mode
        if mode == "convergent":
            tend = 0.7
        elif mode == "divergent":
            tend = 0.2
        elif mode == "maintain":
            tend = 0.0
        else:
            tend = random.uniform(0.3, 0.8)
        return SpeakerAgent(name, result, tend)

    def _extract_meta(self, agent: SpeakerAgent) -> dict:
        r = agent.result
        return {
            "name": agent.name,
            "persona_id": r.demographic.persona_id,
            "persona_description": r.demographic.persona_description,
            "archetype_id": r.archetype.id,
            "archetype_name": r.archetype.name,
            "archetype_name_zh": r.archetype.name_zh,
            "region": r.demographic.region_label,
            "profession": r.demographic.profession_label,
            "L2_proficiency": r.demographic.L2_proficiency_label,
            "language_mode": r.language_mode.level,
            "effective_cmi": round(r.language_mode.effective_cmi, 3),
            "accommodation_tendency": agent.tendency,
        }

    @staticmethod
    def _clean_output(text: str, name: str) -> str:
        """清理 LLM 输出：去掉角色标记、元描述、引号包裹"""
        text = text.strip()
        for prefix in [f"{name}：", f"{name}:", f"Speaker {name}:",
                       "好的，", "以下是", "当然，", "Sure, ", "Okay, "]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith('"') and text.endswith('"')):
            text = text[1:-1].strip()
        return text

    def generate_one(self, idx: int) -> Optional[DialogueOutput]:
        """生成一段完整对话"""

        # 1. 采样 + 创建 Agent
        res_a, res_b = self._sample_pair()
        agent_a = self._make_agent("A", res_a)
        agent_b = self._make_agent("B", res_b)

        # 2. 共享上下文
        topic_id = res_a.situation.topic
        topic_label = res_a.situation.topic_label
        formality = res_a.situation.formality
        formality_desc = FORMALITY_DESCRIPTIONS.get(formality, "轻松随意")
        relationship = res_a.situation.interlocutor_label
        rel_person = RELATIONSHIP_TO_PERSON.get(relationship, relationship)

        # 3. MCP 话题信息获取
        info_snippets = self.topic_router.fetch(
            topic_id, topic_label, self.config.max_info_snippets
        )
        topic_info_text = self.topic_router.format_for_prompt(info_snippets)
        if info_snippets:
            logger.info(
                f"对话 {idx}: 获取到 {len(info_snippets)} 条 {topic_id} 信息"
            )

        # 4. 逐轮生成
        dlg_id = f"DLG_{hashlib.md5(f'{self.config.seed}_{idx}_{time.time()}'.encode()).hexdigest()[:12]}"
        turns: list[DialogueTurn] = []
        history: list[dict] = []
        self.accommodation.reset()

        for turn_num in range(1, self.config.turns_per_dialogue + 1):
            agent = agent_a if turn_num % 2 == 1 else agent_b
            other = "B" if turn_num % 2 == 1 else "A"

            # 适应性指令
            acc_text = self.accommodation.get_instruction(
                agent.result.archetype.CMI_range, other, agent.tendency
            )

            # Self-Example Bank few-shot
            few_shot = []
            if self.config.num_few_shot > 0 and self.example_bank.total > 0:
                few_shot = self.example_bank.retrieve(
                    agent.result.archetype.id, topic_id, self.config.num_few_shot
                )

            # 构建 prompt
            user_prompt = agent.build_turn_prompt(
                history=history,
                turn_num=turn_num,
                total_turns=self.config.turns_per_dialogue,
                topic_label=topic_label,
                formality_desc=formality_desc,
                relationship_desc=rel_person,
                topic_info_text=topic_info_text,
                accommodation_text=acc_text,
                few_shot_examples=few_shot,
            )

            # 带重试的生成 + 评估
            best_text, best_score, best_eval = "", 0.0, None

            for retry in range(self.config.max_retries):
                try:
                    raw = self.llm.chat(
                        agent.system_prompt, user_prompt,
                        self.config.temperature, self.config.max_tokens,
                    )
                except Exception as e:
                    logger.warning(f"LLM 失败 (retry {retry}): {e}")
                    continue

                cleaned = self._clean_output(raw, agent.name)
                if not cleaned:
                    continue

                ev = self.evaluator.evaluate(
                    text=cleaned,
                    archetype_id=agent.result.archetype.id,
                    archetype_cmi_range=agent.result.archetype.CMI_range,
                    formality=formality,
                    region=agent.result.demographic.region,
                    domain_words=agent.result.situation.domain_words,
                )

                if ev.final_score > best_score:
                    best_text, best_score, best_eval = cleaned, ev.final_score, ev

                if ev.final_score >= self.config.min_turn_score:
                    break
                self.stats["retries"] += 1

            if not best_text or best_eval is None:
                self.stats["failed_turns"] += 1
                continue

            # 更新状态
            self.accommodation.observe(agent.name, best_eval.analysis.cmi)
            turns.append(DialogueTurn(
                turn_number=turn_num, speaker_name=agent.name,
                text=best_text, eval_score=best_score,
                eval_decision=best_eval.decision,
                cmi=round(best_eval.analysis.cmi, 3),
                num_switches=len(best_eval.analysis.switch_points),
            ))
            history.append({"speaker": agent.name, "text": best_text})
            self.stats["turns"] += 1

            # 高分入库
            if best_score >= self.config.example_bank_min_score:
                prev_ctx = history[-2]["text"][:100] if len(history) >= 2 else ""
                self.example_bank.add(
                    agent.result.archetype.id, topic_id,
                    best_text, best_score, best_eval.analysis.cmi, prev_ctx,
                )
                self.stats["banked"] += 1

        if not turns:
            return None

        avg = sum(t.eval_score for t in turns) / len(turns)
        self.stats["dialogues"] += 1

        return DialogueOutput(
            dialogue_id=dlg_id, turns=turns,
            speaker_a=self._extract_meta(agent_a),
            speaker_b=self._extract_meta(agent_b),
            topic=topic_label, topic_id=topic_id,
            formality=formality, relationship=relationship,
            language_pair=list(self.config.language_pair),
            avg_score=round(avg, 2), total_turns=len(turns),
            injected_info=[
                {"title": s.title, "source": s.source, "content": s.content[:150]}
                for s in info_snippets
            ],
        )

    def _to_dict(self, d: DialogueOutput) -> dict:
        return {
            "dialogue_id": d.dialogue_id,
            "topic": d.topic, "topic_id": d.topic_id,
            "formality": d.formality, "relationship": d.relationship,
            "language_pair": d.language_pair,
            "avg_score": d.avg_score, "total_turns": d.total_turns,
            "speaker_a": d.speaker_a, "speaker_b": d.speaker_b,
            "injected_info": d.injected_info,
            "turns": [
                {"turn": t.turn_number, "speaker": t.speaker_name,
                 "text": t.text, "score": t.eval_score,
                 "decision": t.eval_decision, "cmi": t.cmi,
                 "switches": t.num_switches}
                for t in d.turns
            ],
        }

    def run(self):
        """批量生成"""
        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始生成 {self.config.num_dialogues} 段对话")
        logger.info(f"模型: {self.config.model} | 每段 {self.config.turns_per_dialogue} 轮")
        logger.info(f"Self-Example Bank: {self.example_bank.total} 条已有示例")

        with open(out, "w", encoding="utf-8") as f:
            for i in range(self.config.num_dialogues):
                try:
                    dlg = self.generate_one(i)
                except Exception as e:
                    logger.error(f"对话 {i} 异常: {e}")
                    continue
                if dlg is None:
                    continue

                f.write(json.dumps(self._to_dict(dlg), ensure_ascii=False) + "\n")
                f.flush()

                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        f"[{i+1}/{self.config.num_dialogues}] "
                        f"turns={self.stats['turns']} "
                        f"retries={self.stats['retries']} "
                        f"banked={self.stats['banked']} "
                        f"score={dlg.avg_score:.1f}"
                    )

        logger.info("=" * 50)
        logger.info("完成！统计：")
        for k, v in self.stats.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"  example_bank: {self.example_bank.total}")
        logger.info(f"  output: {out}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Multi-Turn Dialogue Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-dialogues", type=int, default=100)
    parser.add_argument("--turns-per-dialogue", type=int, default=6)
    parser.add_argument("--language-pair", nargs=2, default=["zh", "en"])
    parser.add_argument("--api-base", nargs="+",
                        default=["http://localhost:8000/v1"],
                        help="vLLM 端点（可指定多个做负载均衡）")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B",
                        help="模型名（需与 vLLM 启动时一致）")
    parser.add_argument("--disable-thinking", action="store_true", default=True,
                        help="关闭 Qwen3 thinking mode（默认开启）")
    parser.add_argument("--enable-thinking", dest="disable_thinking",
                        action="store_false",
                        help="启用 Qwen3 thinking mode")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-turn-score", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--accommodation", default="mixed",
                        choices=["convergent", "divergent", "maintain", "mixed"])
    parser.add_argument("--example-bank", default="")
    parser.add_argument("--example-bank-min-score", type=float, default=8.0)
    parser.add_argument("--num-few-shot", type=int, default=2)
    parser.add_argument("--provider-config", default="")
    parser.add_argument("--max-info-snippets", type=int, default=3)
    parser.add_argument("--output", default="output/dialogues.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = GenerationConfig(
        num_dialogues=args.num_dialogues,
        turns_per_dialogue=args.turns_per_dialogue,
        language_pair=tuple(args.language_pair),
        api_bases=args.api_base, api_key=args.api_key,
        model=args.model, disable_thinking=args.disable_thinking,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_turn_score=args.min_turn_score,
        max_retries=args.max_retries,
        accommodation_mode=args.accommodation,
        example_bank_path=args.example_bank,
        example_bank_min_score=args.example_bank_min_score,
        num_few_shot=args.num_few_shot,
        provider_config_path=args.provider_config,
        max_info_snippets=args.max_info_snippets,
        output_path=args.output, seed=args.seed,
    )

    DialogueGenerator(config).run()


if __name__ == "__main__":
    main()
