"""
SwitchLingua 2.0 — Multi-Turn Dialogue CS Data Generator

Architecture:
  Topic Information (MCP) + Giles CAT Accommodation + Rule-Based Evaluation

  1. TopicRouter 根据话题调用对应 API 获取真实信息（新闻/论文/知识）
  2. 两个 SpeakerAgent 轮流生成，每轮注入话题信息 + 对话历史
  3. AccommodationController 基于 Giles (1991) CAT 动态调整 CS 行为
  4. 每轮经 rule-based 评估器打分，不合格自动重试

Usage:
  # 启动 vLLM 服务后运行
  python dialogue_generator.py \\
      --num-dialogues 100 \\
      --turns-per-dialogue 6 \\
      --api-base http://localhost:8001/v1 \\
      --model /data/models/Qwen3.5-122B-A10B-FP8 \\
      --output output/dialogues.jsonl
"""

import sys
import re
import json
import random
import time
import hashlib
import logging
import argparse
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# stage1_infrastructure on Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1_infrastructure"))

from sampling import ContextualSampler, SamplingResult
from prompt_generator import (
    CS_BEHAVIOR_TEMPLATE_MAP,
    PROFICIENCY_DESCRIPTIONS,
    FORMALITY_DESCRIPTIONS,
    RELATIONSHIP_TO_PERSON,
)
from evaluator_agents import RuleBasedEvaluatorPipeline
from language_config import LanguagePairConfig

from topic_information import TopicRouter, InformationSnippet

logger = logging.getLogger("dialogue_generator")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class DialogueTurn:
    turn_number: int
    speaker_name: str
    text: str
    eval_score: float
    eval_decision: str
    cmi: float
    num_switches: int

@dataclass
class DialogueOutput:
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
    injected_info: list[dict]

@dataclass
class GenerationConfig:
    num_dialogues: int = 100
    turns_per_dialogue: int = 6
    lang_pair: str = "zh_en"
    # LLM
    api_bases: list = field(default_factory=lambda: ["http://localhost:8001/v1"])
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3.5-122B-A10B-FP8"
    disable_thinking: bool = True
    temperature: float = 0.85
    max_tokens: int = 2048
    # Quality
    min_turn_score: float = 3.0
    max_retries: int = 5
    # Accommodation
    accommodation_mode: str = "mixed"
    # Topic Information
    provider_config_path: str = ""
    max_info_snippets: int = 3
    # Output
    output_path: str = "output/dialogues.jsonl"
    seed: int = 42


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    """
    vLLM client with multi-endpoint load balancing.
    Handles Qwen3/3.5 thinking mode suppression.
    """

    def __init__(self, api_bases: list[str], api_key: str = "EMPTY",
                 model: str = "", disable_thinking: bool = True):
        self.api_bases = [b.rstrip("/") for b in api_bases]
        self.api_key = api_key
        self.model = model
        self.disable_thinking = disable_thinking
        self._call_count = 0
        import requests
        self._session = requests.Session()

    def _next_endpoint(self) -> str:
        endpoint = self.api_bases[self._call_count % len(self.api_bases)]
        self._call_count += 1
        return endpoint

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.85, max_tokens: int = 512) -> str:
        endpoint = self._next_endpoint()

        # Qwen3/3.5 thinking mode: suppress via /no_think in prompt + extra_body + stop token
        if self.disable_thinking:
            if "/no_think" not in system_prompt:
                system_prompt = system_prompt + "\n/no_think"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
        }

        # vLLM: pass chat_template_kwargs at top level (not nested in extra_body)
        # This tells Qwen3.5's chat template to disable thinking mode
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            resp = self._session.post(
                f"{endpoint}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=180,
            )
        except Exception as e:
            raise ConnectionError(
                f"API connection failed [{endpoint}]: {type(e).__name__}: {e}"
            )

        if resp.status_code != 200:
            error_detail = ""
            try:
                error_detail = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                error_detail = resp.text[:300]
            raise RuntimeError(
                f"API error [{endpoint}] HTTP {resp.status_code}: {error_detail}"
            )

        try:
            content = resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected API response format: {resp.text[:300]}"
            )

        # Return raw content — cleaning is done by _clean_output()
        if not content:
            raise RuntimeError("Model returned empty content")

        return content


# ============================================================
# Accommodation Controller (Giles 1991 CAT)
# ============================================================

class AccommodationController:
    """
    CS dynamic accommodation based on Communication Accommodation Theory.
    Tracks partner's CMI trend and generates natural language instructions.
    """

    INSTRUCTIONS = {
        "strong_up": "Your partner has been using a lot of English. You naturally increase your English usage too.",
        "up": "Your partner used more English just now. You also naturally add a bit more English.",
        "down": "Your partner used less English just now. You also naturally reduce your English.",
        "strong_down": "Your partner barely uses English. You consciously reduce your English to match.",
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
    One speaker in the dialogue, with persona + archetype -> system prompt.
    """

    def __init__(self, name: str, sampling_result: SamplingResult,
                 accommodation_tendency: float = 0.5,
                 lang_config: Optional[LanguagePairConfig] = None):
        self.name = name
        self.result = sampling_result
        self.tendency = accommodation_tendency
        self.lang_config = lang_config
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        r = self.result
        lc = self.lang_config
        persona = r.demographic.persona_description

        # Proficiency description: prefer lang_config, fallback to hardcoded
        if lc and lc.proficiency_descriptions:
            prof_map = {k: v.format(l2_name=lc.l2_name)
                        for k, v in lc.proficiency_descriptions.items()}
        else:
            prof_map = PROFICIENCY_DESCRIPTIONS
        proficiency = prof_map.get(r.demographic.L2_proficiency, "")

        # Role line
        if lc and lc.role_template:
            role = lc.role_template.format(persona=persona, proficiency=proficiency) + "\n"
        else:
            role = f"你是一个{persona}。{proficiency}\n"

        # Domain context
        l2_name = lc.l2_name if lc else "英文"
        domain_words = r.situation.domain_words
        topic = r.situation.topic_label
        if domain_words:
            domain_ctx = f"在谈论{topic}相关话题时，会自然地使用{l2_name}词汇如{'、'.join(domain_words[:5])}等"
        else:
            domain_ctx = f"在谈论{topic}相关话题时，偶尔会使用一些{l2_name}词汇"

        # CS behavior template: prefer lang_config
        if lc and lc.cs_behavior_templates:
            cs_templates = lc.cs_behavior_templates
        else:
            cs_templates = CS_BEHAVIOR_TEMPLATE_MAP
        cs_template = cs_templates.get(r.archetype.id, cs_templates.get("ARC_01", ""))
        cs_part = cs_template.format(
            domain_context=domain_ctx,
            l1_name=lc.l1_name if lc else "中文",
            l2_name=l2_name,
        )
        level = r.language_mode.description

        base = f"{role}{cs_part}\n\n【语言混合程度】{level}"
        base += (
            "\n\n【重要】你只需要输出角色说的话，用 <reply></reply> 标签包裹。"
            "禁止输出分析、思考过程或任何解释。直接输出对话内容。"
        )
        return base.strip()

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
    ) -> str:
        """
        Build user prompt for one turn.
        Injects: topic information + dialogue history + accommodation instruction.
        """
        parts = []
        lc = self.lang_config

        # Use English frame instructions for non-CJK language pairs
        # (LLM understands English instructions universally)
        use_en = lc and lc.l1_script not in ("cjk", "kana_kanji")

        if use_en:
            parts.append(
                f"You are having a conversation about '{topic_label}' with {relationship_desc}. "
                f"The tone is {formality_desc}."
            )
        else:
            parts.append(
                f"你正在和一个{relationship_desc}进行关于「{topic_label}」的对话。"
                f"对话氛围{formality_desc}。"
            )

        # Topic information (MCP)
        if topic_info_text:
            parts.append("")
            parts.append(topic_info_text)

        # Dialogue history
        if history:
            parts.append("")
            parts.append("Dialogue so far:" if use_en else "以下是目前的对话内容：")
            for h in history:
                parts.append(f"{h['speaker']}：{h['text']}")

        # Accommodation instruction
        if accommodation_text:
            parts.append("")
            parts.append(accommodation_text)

        # Generation instruction
        parts.append("")
        if turn_num == 1:
            if use_en:
                parts.append(
                    f"As {self.name}, start the conversation. "
                    f"Use your natural way of speaking."
                )
            else:
                parts.append(
                    f"请你作为 {self.name} 开始这段对话。"
                    f"用你自然的说话方式发起话题。"
                )
        elif turn_num == total_turns:
            if use_en:
                parts.append(
                    f"As {self.name}, respond and naturally wrap up the conversation."
                )
            else:
                parts.append(
                    f"请你作为 {self.name} 回应对方，并自然地结束对话。"
                )
        else:
            if use_en:
                parts.append(
                    f"As {self.name}, respond naturally. "
                    f"You may answer, ask follow-up questions, share opinions, or extend the topic."
                )
            else:
                parts.append(
                    f"请你作为 {self.name} 自然地回应对方。"
                    f"可以回答、追问、表达看法或延伸话题。"
                )

        # Diverse example pool from lang_config or fallback
        lc = self.lang_config
        if lc and lc.example_dialogues:
            examples = lc.example_dialogues
        else:
            examples = [
                "嗯那个meeting开完了，整个人有点exhausted，想找个cafe坐一会。",
                "你有没有试过那个app？我觉得UI design还不错，就是loading有点慢。",
                "昨天那个seminar讲的topic蛮interesting的，不过slides太多了看得我头晕。",
            ]
        example = random.choice(examples)

        # Generation requirements from lang_config or fallback
        if lc and lc.generation_requirements:
            req_text = lc.generation_requirements.format(
                l1_name=lc.l1_name, l2_name=lc.l2_name)
        else:
            req_text = (
                "- 必须中英文混合说话，在句子中自然地嵌入英文单词或短语\n"
                "- 【重要】长度严格控制在2-3句话，总字数不超过80字，宁短勿长\n"
                "- 可以包含犹豫词和自我修正\n"
                "- 不要反复提及同一个话题信息的名称"
            )

        if use_en:
            parts.append(
                f"\nRequirements:\n{req_text}\n"
                f"- Reference style: '{example}'\n\n"
                "[Output format] Only output what you say, wrapped in <reply> tags. "
                "Do NOT output any analysis, thinking process, or explanation.\n"
                f"Example: <reply>{example}</reply>"
            )
        else:
            parts.append(
                f"\n要求：\n{req_text}\n"
                f"- 参考风格：'{example}'\n\n"
                "【输出格式】只输出你说的话，用 <reply> 标签包裹，不要输出任何分析、解释或思考过程。\n"
                f"例如：<reply>{example}</reply>"
            )

        return "\n".join(parts)


# ============================================================
# Terminal Display
# ============================================================

def print_dialogue_header(idx: int, topic: str, relationship: str,
                          agent_a: SpeakerAgent, agent_b: SpeakerAgent):
    """Print dialogue header to terminal"""
    print(f"\n{'='*70}")
    print(f"  Dialogue #{idx + 1}  |  Topic: {topic}  |  Relationship: {relationship}")
    print(f"  A: {agent_a.result.demographic.persona_description} "
          f"[{agent_a.result.archetype.name_zh}]")
    print(f"  B: {agent_b.result.demographic.persona_description} "
          f"[{agent_b.result.archetype.name_zh}]")
    print(f"{'='*70}")


def print_turn(turn: DialogueTurn):
    """Print one turn to terminal"""
    score_color = "\033[92m" if turn.eval_score >= 7.0 else (
        "\033[93m" if turn.eval_score >= 5.0 else "\033[91m"
    )
    reset = "\033[0m"
    print(f"\n  [{turn.speaker_name}] {turn.text}")
    print(f"       {score_color}score={turn.eval_score:.1f}{reset}  "
          f"CMI={turn.cmi:.3f}  switches={turn.num_switches}")


def print_turn_failure(speaker_name: str, turn_num: int, reason: str,
                       raw_text: str = ""):
    """Print turn failure to terminal, including the raw text for debugging"""
    print(f"\n  [{speaker_name}] \033[91m(Turn {turn_num} FAILED: {reason})\033[0m")
    if raw_text:
        preview = raw_text[:150].replace('\n', ' ')
        print(f"       \033[90mRaw: {preview}{'...' if len(raw_text) > 150 else ''}\033[0m")


def print_dialogue_footer(dialogue: DialogueOutput):
    """Print dialogue summary"""
    print(f"\n  {'- '*35}")
    print(f"  Avg Score: {dialogue.avg_score:.1f}  |  "
          f"Turns: {dialogue.total_turns}/{len(dialogue.turns)}")


# ============================================================
# Main Generator
# ============================================================

class DialogueGenerator:
    """
    Multi-turn CS dialogue generator.

    Flow per dialogue:
    1. Sample two speakers with different personas/archetypes
    2. Fetch topic information via MCP (TopicRouter)
    3. Generate turn-by-turn:
       a. Build prompt = system(persona) + user(topic_info + history + accommodation)
       b. Call LLM
       c. Rule-based evaluation, retry if below threshold
       d. Display turn in terminal
       e. Update accommodation state
    4. Write to JSONL
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        random.seed(config.seed)

        # Load language pair configuration
        self.lang_config = LanguagePairConfig.load(config.lang_pair)
        logger.info(f"Loaded language pair: {self.lang_config.display_name} ({config.lang_pair})")

        self.sampler = ContextualSampler()
        self.llm = LLMClient(
            config.api_bases, config.api_key, config.model,
            config.disable_thinking,
        )
        self.evaluator = RuleBasedEvaluatorPipeline()
        self.accommodation = AccommodationController(config.accommodation_mode)

        # Topic Information Router
        provider_cfg = config.provider_config_path or None
        self.topic_router = TopicRouter(provider_cfg)

        self.stats = Counter()

    def _sample_pair(self) -> tuple[SamplingResult, SamplingResult]:
        """Sample two speakers with different profiles"""
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
        return SpeakerAgent(name, result, tend, lang_config=self.lang_config)

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
        """
        Clean LLM output:
        1. Extract <reply> tags (only reliable method)
        2. Strip <think> XML blocks
        3. Reject anything that looks like CoT / prompt echo
        """
        text = text.strip()

        # 1. Strip <think> blocks first
        if "<think>" in text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # 2. Extract <reply> tag content — the ONLY reliable strategy
        reply_match = re.search(r'<reply>(.*?)</reply>', text, flags=re.DOTALL)
        if reply_match:
            text = reply_match.group(1).strip()
        elif '<reply>' in text:
            # Unclosed <reply> tag (truncated by max_tokens) — extract content after it
            text = text.split('<reply>', 1)[1].strip()
        else:
            # No <reply> tag: reject if it looks like CoT or prompt echo
            cot_signals = [
                "Analyze the Request", "Thinking Process", "Determine Content",
                "Generate Response", "Final Output", "Let me think",
                "Here's my response", "**Analyze", "**Role:", "**Task:",
                "self-correction", "self-corrections",
                "请你作为", "以下是目前的对话",
            ]
            is_cot = (
                any(sig in text for sig in cot_signals)
                or bool(re.match(r'^\d+\.\s+', text))
                or len(text) > 200  # real dialogue should be < 100 chars
            )
            if is_cot:
                return ""
            # If text is short, has Chinese, and no CoT signals — it might
            # be a clean response without <reply> tags. Allow it through.

        # 3. Remove remaining markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* -> italic

        # 4. Remove role markers (multi-language)
        for prefix in [f"{name}：", f"{name}:", f"Speaker {name}:",
                       f"说话人{name}：", f"A：", f"B：", f"A:", f"B:",
                       "好的，", "以下是", "当然，", "没问题，",
                       "Sure, ", "Here's ", "Okay, ", "Of course, ",
                       "Here is my response:", "My response:",
                       "Bien, ", "Bueno, ", "Vale, ", "D'accord, ",
                       "はい、", "えーと、", "अच्छा, ", "Baik, "]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 5. Remove wrapping quotes
        if len(text) > 2:
            if (text[0] == '"' and text[-1] == '"') or \
               (text[0] == '\u201c' and text[-1] == '\u201d'):
                text = text[1:-1].strip()

        # 6. Clean up whitespace
        text = re.sub(r'\n+', ' ', text).strip()
        text = re.sub(r'\s{2,}', ' ', text)

        # 7. Reject if still too long (soft target in prompt, hard limit here)
        if len(text) > 200:
            return ""

        return text

    def generate_one(self, idx: int) -> Optional[DialogueOutput]:
        """Generate one complete dialogue"""

        # 1. Sample speakers
        res_a, res_b = self._sample_pair()
        agent_a = self._make_agent("A", res_a)
        agent_b = self._make_agent("B", res_b)

        # 2. Shared context
        topic_id = res_a.situation.topic
        topic_label = res_a.situation.topic_label
        formality = res_a.situation.formality
        formality_desc = FORMALITY_DESCRIPTIONS.get(formality, "轻松随意")
        relationship = res_a.situation.interlocutor_label
        rel_person = RELATIONSHIP_TO_PERSON.get(relationship, relationship)

        # 3. Fetch topic information via MCP
        info_snippets = []
        try:
            info_snippets = self.topic_router.fetch(
                topic_id, topic_label, self.config.max_info_snippets
            )
        except Exception as e:
            logger.warning(f"Topic info fetch failed for '{topic_id}': {e}")

        topic_info_text = self.topic_router.format_for_prompt(info_snippets)

        # Print dialogue header
        print_dialogue_header(idx, topic_label, relationship, agent_a, agent_b)
        if info_snippets:
            print(f"  Topic info: {len(info_snippets)} snippets from "
                  f"{', '.join(set(s.source for s in info_snippets))}")

        # 4. Turn-by-turn generation
        dlg_id = f"DLG_{hashlib.md5(f'{self.config.seed}_{idx}_{time.time()}'.encode()).hexdigest()[:12]}"
        turns: list[DialogueTurn] = []
        history: list[dict] = []
        self.accommodation.reset()

        for turn_num in range(1, self.config.turns_per_dialogue + 1):
            agent = agent_a if turn_num % 2 == 1 else agent_b
            other = "B" if turn_num % 2 == 1 else "A"

            # Accommodation instruction
            acc_text = self.accommodation.get_instruction(
                agent.result.archetype.CMI_range, other, agent.tendency
            )

            # Build prompt
            user_prompt = agent.build_turn_prompt(
                history=history,
                turn_num=turn_num,
                total_turns=self.config.turns_per_dialogue,
                topic_label=topic_label,
                formality_desc=formality_desc,
                relationship_desc=rel_person,
                topic_info_text=topic_info_text,
                accommodation_text=acc_text,
            )

            # Generate with retries
            best_text, best_score, best_eval = "", 0.0, None
            last_error = ""
            last_raw = ""

            for retry in range(self.config.max_retries):
                try:
                    raw = self.llm.chat(
                        agent.system_prompt, user_prompt,
                        self.config.temperature, self.config.max_tokens,
                    )
                except ConnectionError as e:
                    last_error = f"Connection: {e}"
                    logger.error(f"Turn {turn_num} retry {retry+1}: {last_error}")
                    time.sleep(2)  # wait before retry on connection error
                    continue
                except RuntimeError as e:
                    last_error = f"API: {e}"
                    logger.error(f"Turn {turn_num} retry {retry+1}: {last_error}")
                    continue
                except Exception as e:
                    last_error = f"Unexpected: {type(e).__name__}: {e}"
                    logger.error(f"Turn {turn_num} retry {retry+1}: {last_error}")
                    continue

                last_raw = raw
                cleaned = self._clean_output(raw, agent.name)
                if not cleaned:
                    raw_preview = raw[:120].replace('\n', ' ')
                    last_error = f"Empty output after cleaning | Raw: {raw_preview}"
                    logger.warning(f"Turn {turn_num} retry {retry+1}: {last_error}")
                    self.stats["retries"] += 1
                    continue

                # Rule-based evaluation
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
                else:
                    last_error = (
                        f"Score {ev.final_score:.1f} < {self.config.min_turn_score} | "
                        f"Violations: {'; '.join(v for cr in ev.checker_results.values() for v in cr.violations)}"
                    )
                    self.stats["retries"] += 1

            if not best_text or best_eval is None:
                self.stats["failed_turns"] += 1
                print_turn_failure(agent.name, turn_num, last_error, last_raw)
                continue

            # Update accommodation state
            self.accommodation.observe(agent.name, best_eval.analysis.cmi)

            turn = DialogueTurn(
                turn_number=turn_num, speaker_name=agent.name,
                text=best_text, eval_score=best_score,
                eval_decision=best_eval.decision,
                cmi=round(best_eval.analysis.cmi, 3),
                num_switches=len(best_eval.analysis.switch_points),
            )
            turns.append(turn)
            history.append({"speaker": agent.name, "text": best_text})
            self.stats["turns"] += 1

            # Print turn to terminal
            print_turn(turn)

        if not turns:
            print(f"\n  \033[91mDialogue #{idx+1} FAILED: no valid turns\033[0m")
            return None

        avg = sum(t.eval_score for t in turns) / len(turns)
        self.stats["dialogues"] += 1

        dialogue = DialogueOutput(
            dialogue_id=dlg_id, turns=turns,
            speaker_a=self._extract_meta(agent_a),
            speaker_b=self._extract_meta(agent_b),
            topic=topic_label, topic_id=topic_id,
            formality=formality, relationship=relationship,
            language_pair=[self.lang_config.l1_code, self.lang_config.l2_code],
            avg_score=round(avg, 2), total_turns=len(turns),
            injected_info=[
                {"title": s.title, "source": s.source, "content": s.content[:150]}
                for s in info_snippets
            ],
        )

        print_dialogue_footer(dialogue)
        return dialogue

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
        """Run batch generation"""
        out = Path(self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"  SwitchLingua 2.0 Dialogue Generator")
        print(f"  Model: {self.config.model}")
        print(f"  Endpoints: {', '.join(self.config.api_bases)}")
        print(f"  Dialogues: {self.config.num_dialogues} x {self.config.turns_per_dialogue} turns")
        print(f"  Output: {out}")
        print(f"{'#'*70}")

        with open(out, "w", encoding="utf-8") as f:
            for i in range(self.config.num_dialogues):
                try:
                    dlg = self.generate_one(i)
                except Exception as e:
                    logger.error(f"Dialogue {i} exception: {e}")
                    traceback.print_exc()
                    continue
                if dlg is None:
                    continue

                f.write(json.dumps(self._to_dict(dlg), ensure_ascii=False) + "\n")
                f.flush()

        # Final stats
        print(f"\n{'#'*70}")
        print(f"  Generation Complete!")
        print(f"  Dialogues: {self.stats['dialogues']}")
        print(f"  Turns: {self.stats['turns']}")
        print(f"  Retries: {self.stats['retries']}")
        print(f"  Failed turns: {self.stats['failed_turns']}")
        print(f"  Output: {out}")
        print(f"{'#'*70}\n")


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
    parser.add_argument("--lang-pair", default="zh_en",
                        help="Language pair config ID (e.g. zh_en, yue_en, fr_en)")
    parser.add_argument("--api-base", nargs="+",
                        default=["http://localhost:8001/v1"],
                        help="vLLM endpoint(s) for load balancing")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-FP8",
                        help="Model name or local path")
    parser.add_argument("--disable-thinking", action="store_true", default=True,
                        help="Disable Qwen3/3.5 thinking mode")
    parser.add_argument("--enable-thinking", dest="disable_thinking",
                        action="store_false")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-turn-score", type=float, default=3.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--accommodation", default="mixed",
                        choices=["convergent", "divergent", "maintain", "mixed"])
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
        lang_pair=args.lang_pair,
        api_bases=args.api_base, api_key=args.api_key,
        model=args.model, disable_thinking=args.disable_thinking,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_turn_score=args.min_turn_score,
        max_retries=args.max_retries,
        accommodation_mode=args.accommodation,
        provider_config_path=args.provider_config,
        max_info_snippets=args.max_info_snippets,
        output_path=args.output, seed=args.seed,
    )

    DialogueGenerator(config).run()


if __name__ == "__main__":
    main()
