"""
SwitchLingua 2.0 — Stage 1 Layer 3: 情境化参数采样器 (v2)
Contextual Parameter Sampler for Generation-Time Instantiation

v2 改动：
- 用 Persona Templates 替代独立维度采样，确保人物各维度的合理性
- 原型采样受 persona 的 compatible_archetypes 约束
- 话题采样受 persona 的 natural_topics 约束
- 关系采样受 persona 的 compatible_relationships 约束
- 正式度由关系的 default_formality 决定（不再独立采样）
- 领域词按话题 × 职业交叉选取
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class SampledArchetype:
    """采样得到的原型"""
    id: str
    name: str
    name_zh: str
    CMI_range: list[float]
    dominant_switch_type: str
    switch_triggers: list[str]
    L2_span_length: str
    example_description: str


@dataclass
class SampledDemographic:
    """采样得到的人口学背景"""
    persona_id: str
    persona_description: str
    region: str
    region_label: str
    age_group: str
    profession: str
    profession_label: str
    L2_proficiency: str
    L2_proficiency_label: str


@dataclass
class SampledSituation:
    """采样得到的情境"""
    topic: str
    topic_label: str
    L2_affinity: str
    formality: str
    formality_label: str
    dialogue_type: str
    interlocutor_relationship: str
    interlocutor_label: str
    domain_words: list[str]


@dataclass
class LanguageMode:
    """Language Mode Controller 输出"""
    level: str
    description: str
    effective_cmi: float


@dataclass
class SamplingResult:
    """完整的采样结果"""
    archetype: SampledArchetype
    demographic: SampledDemographic
    situation: SampledSituation
    language_mode: LanguageMode
    generation_metadata: dict = field(default_factory=dict)


# ============================================================
# 采样器
# ============================================================

class ContextualSampler:
    """
    三层画像架构的情境化参数采样器 (v2)。

    采样流程：
    1. 按权重采样一个 Persona Template（人物已内部协调）
    2. 从 persona 的 compatible_archetypes 中采样一个原型
    3. 从 persona 的 natural_topics 中采样话题
    4. 从 persona 的 compatible_relationships 中采样关系
    5. 关系决定正式度
    6. 话题 × 职业 交叉决定领域词
    7. 原型 + 情境 → Language Mode
    """

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent
        else:
            config_dir = Path(config_dir)

        self.archetypes_data = self._load_yaml(config_dir / "archetypes.yaml")
        self.calibration = self._load_json(config_dir / "corpus_calibration.json")
        self.pools = self._load_yaml(config_dir / "background_pools.yaml")

        self._build_indices()

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_indices(self):
        """构建查找索引"""
        # 原型 index
        self._archetypes = {
            a["id"]: a for a in self.archetypes_data["archetypes"]
        }
        # 全局原型采样权重（用于计算 persona 内的相对权重）
        self._arc_global_weights = self.archetypes_data["sampling_weights"]

        # Persona templates
        self._personas = self.pools["persona_templates"]
        self._persona_weights = [p["weight"] for p in self._personas]

        # 话题 index
        self._topics = {
            t["id"]: t for t in self.pools["situation_pools"]["topics"]
        }
        self._topic_weights = {
            t["id"]: t["weight"] for t in self.pools["situation_pools"]["topics"]
        }

        # 关系 index
        self._relationships = {
            r["id"]: r for r in self.pools["situation_pools"]["interlocutor_relationships"]
        }

        # 正式度 index
        self._formalities = {
            f["id"]: f for f in self.pools["situation_pools"]["formality_levels"]
        }

        # 对话类型
        self._dialogue_types = self.pools["situation_pools"]["dialogue_types"]
        self._dialogue_weights = [d["weight"] for d in self._dialogue_types]

        # 话题领域词
        self._topic_domain_words = self.pools.get("topic_domain_words", {})

    @staticmethod
    def _weighted_choice(items, weights):
        return random.choices(items, weights=weights, k=1)[0]

    def _get_archetype_def(self, arc_id: str) -> dict:
        return self._archetypes[arc_id]

    # ---------- Step 1: 采样 Persona ----------

    def _sample_persona(self) -> dict:
        return self._weighted_choice(self._personas, self._persona_weights)

    # ---------- Step 2: 采样 Archetype (受 persona 约束) ----------

    def _sample_archetype(self, persona: dict) -> SampledArchetype:
        compatible = persona["compatible_archetypes"]

        # 在兼容原型中按全局权重采样
        weights = []
        for arc_id in compatible:
            w = 0.15  # default
            for k, v in self._arc_global_weights.items():
                if arc_id.lower() in k.lower():
                    w = v
                    break
            weights.append(w)

        arc_id = self._weighted_choice(compatible, weights)
        arc_def = self._get_archetype_def(arc_id)

        examples = arc_def.get("example_descriptions", [])
        chosen = random.choice(examples) if examples else {}

        return SampledArchetype(
            id=arc_id,
            name=arc_def["name"],
            name_zh=arc_def["name_zh"],
            CMI_range=arc_def["behavioral_profile"]["CMI_range"],
            dominant_switch_type=arc_def["behavioral_profile"]["dominant_switch_type"],
            switch_triggers=arc_def["behavioral_profile"]["switch_triggers"],
            L2_span_length=arc_def["behavioral_profile"]["L2_span_length"],
            example_description=chosen.get("prompt", ""),
        )

    # ---------- Step 3: 构建 Demographic ----------

    def _build_demographic(self, persona: dict) -> SampledDemographic:
        return SampledDemographic(
            persona_id=persona["id"],
            persona_description=persona["description"],
            region=persona["region"]["id"],
            region_label=persona["region"]["label"],
            age_group=persona["age_group"],
            profession=persona["profession"]["id"],
            profession_label=persona["profession"]["label"],
            L2_proficiency=persona["L2_proficiency"]["id"],
            L2_proficiency_label=persona["L2_proficiency"]["label"],
        )

    # ---------- Step 4: 采样 Situation (受 persona 约束) ----------

    def _get_domain_words(self, topic_id: str, profession_id: str) -> list[str]:
        """按 话题 × 职业 交叉选取领域词"""
        topic_words = self._topic_domain_words.get(topic_id, {})
        if not topic_words:
            return []

        # 检查是否有该职业的专门词汇
        profession_key = f"{profession_id}_profession"
        if profession_key in topic_words:
            return topic_words[profession_key]

        return topic_words.get("general", [])

    def _sample_situation(self, persona: dict, profession_id: str) -> SampledSituation:
        # 话题：从 persona 的 natural_topics 中按全局权重采样
        natural = persona["natural_topics"]
        topic_weights = [self._topic_weights.get(t, 0.1) for t in natural]
        topic_id = self._weighted_choice(natural, topic_weights)
        topic_def = self._topics[topic_id]

        # 关系：从 persona 的 compatible_relationships 中均匀采样
        rel_id = random.choice(persona["compatible_relationships"])
        rel_def = self._relationships[rel_id]

        # 正式度：由关系决定
        formality_id = rel_def["default_formality"]
        formality_def = self._formalities[formality_id]

        # 对话类型：按全局权重
        dialogue = self._weighted_choice(self._dialogue_types, self._dialogue_weights)

        # 领域词：话题 × 职业
        domain_words = self._get_domain_words(topic_id, profession_id)

        return SampledSituation(
            topic=topic_id,
            topic_label=topic_def["label"],
            L2_affinity=topic_def["L2_affinity"],
            formality=formality_id,
            formality_label=formality_def["label"],
            dialogue_type=dialogue["id"],
            interlocutor_relationship=rel_id,
            interlocutor_label=rel_def["label"],
            domain_words=domain_words,
        )

    # ---------- Language Mode Controller ----------

    def compute_language_mode(
        self, archetype: SampledArchetype, situation: SampledSituation
    ) -> LanguageMode:
        cmi_low, cmi_high = archetype.CMI_range
        base_cmi = (cmi_low + cmi_high) / 2

        shift = 0.0

        # 正式度调节
        formality_shifts = {"casual": 0.05, "semi_formal": 0.0, "formal": -0.10}
        shift += formality_shifts.get(situation.formality, 0.0)

        # 话题 L2 亲和度调节
        topic_shifts = self.calibration.get("cmi_distribution", {}).get("by_topic", {})
        topic_data = topic_shifts.get(situation.topic, {})
        shift += topic_data.get("mean_shift", 0.0)

        effective_cmi = max(0.0, min(1.0, base_cmi + shift))

        if effective_cmi < 0.08:
            level, desc = "minimal", "几乎不使用英文，仅限无法翻译的专有名词。一段话中最多1个英文词"
        elif effective_cmi < 0.20:
            level, desc = "light", "偶尔嵌入英文单词或短语，以中文为绝对主体。大约每10个词里有1个英文词"
        elif effective_cmi < 0.35:
            level, desc = "moderate", "适度混合中英文，在术语和特定话题上自然切换。大约每5-6个词里有1个英文词"
        elif effective_cmi < 0.50:
            level, desc = "heavy", "频繁在中英文之间切换，包括句内和句间切换。大约每3-4个词里有1个英文词"
        else:
            level, desc = "dense", "中英文深度融合，两种语言在句内密集交替。大约每2-3个词里有1个英文词"

        return LanguageMode(level=level, description=desc, effective_cmi=effective_cmi)

    # ---------- 主采样方法 ----------

    def sample(self, seed: Optional[int] = None) -> SamplingResult:
        if seed is not None:
            random.seed(seed)

        persona = self._sample_persona()
        archetype = self._sample_archetype(persona)
        demographic = self._build_demographic(persona)
        situation = self._sample_situation(persona, demographic.profession)
        language_mode = self.compute_language_mode(archetype, situation)

        return SamplingResult(
            archetype=archetype,
            demographic=demographic,
            situation=situation,
            language_mode=language_mode,
            generation_metadata={
                "sampler_version": "2.0",
                "persona_id": persona["id"],
                "seed": seed,
            },
        )

    def sample_batch(self, n: int, seed: Optional[int] = None) -> list[SamplingResult]:
        if seed is not None:
            random.seed(seed)
        return [self.sample() for _ in range(n)]


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwitchLingua 2.0 Contextual Sampler v2")
    parser.add_argument("-n", type=int, default=5, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    sampler = ContextualSampler()
    results = sampler.sample_batch(args.n, seed=args.seed)

    output_data = []
    for i, r in enumerate(results):
        entry = {
            "index": i,
            "persona_id": r.demographic.persona_id,
            "persona_description": r.demographic.persona_description,
            "archetype": {
                "id": r.archetype.id,
                "name": r.archetype.name,
                "name_zh": r.archetype.name_zh,
                "CMI_range": r.archetype.CMI_range,
            },
            "demographic": {
                "region": r.demographic.region_label,
                "age_group": r.demographic.age_group,
                "profession": r.demographic.profession_label,
                "L2_proficiency": r.demographic.L2_proficiency_label,
            },
            "situation": {
                "topic": r.situation.topic_label,
                "formality": r.situation.formality_label,
                "dialogue_type": r.situation.dialogue_type,
                "relationship": r.situation.interlocutor_label,
                "domain_words": r.situation.domain_words,
            },
            "language_mode": {
                "level": r.language_mode.level,
                "description": r.language_mode.description,
                "effective_cmi": round(r.language_mode.effective_cmi, 3),
            },
        }
        output_data.append(entry)

    output_json = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Saved {args.n} samples to {args.output}")
    else:
        print(output_json)
