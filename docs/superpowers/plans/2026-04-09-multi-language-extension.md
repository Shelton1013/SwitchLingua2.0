# Multi-Language Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend SwitchLingua 2.0 from Mandarin-English only to 8 language pairs via shared engine + per-language config folders.

**Architecture:** Each language pair is a self-contained config folder under `configs/{pair_id}/`. Core code is parameterized to read all language-specific content from `LanguagePairConfig`. Two detection modes: Unicode (CJK/Devanagari) and Lexicon (dual-Latin).

**Tech Stack:** Python 3.10, PyYAML, existing vLLM inference pipeline

**Spec:** `docs/superpowers/specs/2026-04-09-multi-language-extension-design.md`

---

## Phase 1: Infrastructure + zh_en Extraction

### Task 1: Create LanguagePairConfig dataclass and loader

**Files:**
- Create: `stage1_infrastructure/language_config.py`

- [ ] **Step 1: Create `language_config.py`**

```python
"""
SwitchLingua 2.0 — Language Pair Configuration Loader

Loads all language-specific configuration from configs/{pair_id}/ folder.
Provides a unified LanguagePairConfig object consumed by all downstream modules.
"""

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LanguagePairConfig:
    """Unified configuration for one language pair."""
    # Identity
    pair_id: str
    l1_code: str
    l2_code: str
    l1_name: str
    l2_name: str
    l1_name_en: str
    l2_name_en: str

    # Script & detection
    l1_script: str          # "cjk", "latin", "devanagari", "kana_kanji"
    l2_script: str          # "latin"
    detection_mode: str     # "unicode" or "lexicon"
    l1_unicode_ranges: list[tuple[int, int]]
    l2_unicode_ranges: list[tuple[int, int]]

    # Punctuation
    sentence_endings: str
    l1_punctuation: str

    # Filler words
    filler_words_l1: list[str]
    filler_words_l2: list[str]
    interjections_l1: list[str]

    # Prompt templates
    role_template: str
    cs_behavior_templates: dict[str, str]
    proficiency_descriptions: dict[str, str]
    mixing_level_descriptions: dict[str, str]
    generation_requirements: str
    example_dialogues: list[str]

    # Personas, calibration, evaluation
    personas: list[dict]
    calibration: dict
    evaluation: dict

    # Lexicon (dual-Latin pairs only)
    l1_lexicon: Optional[set[str]] = None

    # Optional overrides
    max_retries_override: Optional[int] = None

    @classmethod
    def load(cls, pair_id: str, configs_root: Optional[str] = None) -> "LanguagePairConfig":
        """Load config from configs/{pair_id}/ folder."""
        if configs_root is None:
            configs_root = Path(__file__).parent.parent / "configs"
        else:
            configs_root = Path(configs_root)

        config_dir = configs_root / pair_id
        if not config_dir.exists():
            available = [d.name for d in configs_root.iterdir() if d.is_dir()]
            raise FileNotFoundError(
                f"Language pair config '{pair_id}' not found at {config_dir}. "
                f"Available: {available}"
            )

        # Load YAML/JSON files
        language = cls._load_yaml(config_dir / "language.yaml")
        prompts = cls._load_yaml(config_dir / "prompts.yaml")
        personas = cls._load_yaml(config_dir / "personas.yaml")
        calibration = cls._load_json(config_dir / "calibration.json")
        evaluation = cls._load_yaml(config_dir / "evaluation.yaml")

        # Parse unicode ranges from lists to tuples
        l1_ranges = [tuple(r) for r in language.get("l1_unicode_ranges", [])]
        l2_ranges = [tuple(r) for r in language.get("l2_unicode_ranges", [])]

        # Load lexicon if exists (dual-Latin pairs)
        l1_lexicon = None
        lexicon_path = config_dir / "l1_lexicon.txt"
        if lexicon_path.exists():
            with open(lexicon_path, "r", encoding="utf-8") as f:
                l1_lexicon = set(
                    line.strip().lower()
                    for line in f if line.strip() and not line.startswith("#")
                )

        return cls(
            pair_id=pair_id,
            l1_code=language["l1_code"],
            l2_code=language["l2_code"],
            l1_name=language["l1_name"],
            l2_name=language["l2_name"],
            l1_name_en=language["l1_name_en"],
            l2_name_en=language["l2_name_en"],
            l1_script=language["l1_script"],
            l2_script=language["l2_script"],
            detection_mode=language.get("detection_mode", "unicode"),
            l1_unicode_ranges=l1_ranges,
            l2_unicode_ranges=l2_ranges,
            sentence_endings=language.get("sentence_endings", "。！？.!?"),
            l1_punctuation=language.get("l1_punctuation", ""),
            filler_words_l1=language.get("filler_words_l1", []),
            filler_words_l2=language.get("filler_words_l2", []),
            interjections_l1=language.get("interjections_l1", []),
            role_template=prompts.get("role_template", "你是一个{persona}。{proficiency}"),
            cs_behavior_templates=prompts.get("cs_behavior_templates", {}),
            proficiency_descriptions=prompts.get("proficiency_descriptions", {}),
            mixing_level_descriptions=prompts.get("mixing_level_descriptions", {}),
            generation_requirements=prompts.get("generation_requirements", ""),
            example_dialogues=prompts.get("example_dialogues", []),
            personas=personas.get("persona_templates", []),
            calibration=calibration,
            evaluation=evaluation,
            l1_lexicon=l1_lexicon,
            max_retries_override=language.get("max_retries_override"),
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def list_available(cls, configs_root: Optional[str] = None) -> list[str]:
        """List all available language pair config IDs."""
        if configs_root is None:
            configs_root = Path(__file__).parent.parent / "configs"
        else:
            configs_root = Path(configs_root)
        if not configs_root.exists():
            return []
        return sorted(
            d.name for d in configs_root.iterdir()
            if d.is_dir() and (d / "language.yaml").exists()
        )
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('stage1_infrastructure/language_config.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add stage1_infrastructure/language_config.py
git commit -m "feat: add LanguagePairConfig dataclass and loader"
```

---

### Task 2: Create configs/zh_en/ — extract from hardcoded values

**Files:**
- Create: `configs/zh_en/language.yaml`
- Create: `configs/zh_en/prompts.yaml`
- Create: `configs/zh_en/personas.yaml`
- Create: `configs/zh_en/calibration.json`
- Create: `configs/zh_en/evaluation.yaml`

- [ ] **Step 1: Create `configs/zh_en/language.yaml`**

Extract from `evaluator_agents.py:112-116` (unicode ranges), `evaluator_agents.py:119` (sentence endings), `evaluator_agents.py:736` (filler words), and `corpus_calibration.json:123-131` (filler words).

```yaml
# SwitchLingua 2.0 — Mandarin-English Language Pair Config
l1_code: "zh"
l2_code: "en"
l1_name: "中文"
l2_name: "英文"
l1_name_en: "Chinese"
l2_name_en: "English"

l1_script: "cjk"
l2_script: "latin"
detection_mode: "unicode"

l1_unicode_ranges:
  - [0x4E00, 0x9FFF]    # CJK Unified Ideographs
  - [0x3400, 0x4DBF]    # CJK Extension A
  - [0xF900, 0xFAFF]    # CJK Compatibility Ideographs

l2_unicode_ranges:
  - [0x0041, 0x005A]    # A-Z
  - [0x0061, 0x007A]    # a-z

sentence_endings: "。！？.!?"
l1_punctuation: "，、；：""''（）—…"

filler_words_l1: ["嗯", "那个", "就是", "然后", "啊", "哎", "对"]
filler_words_l2: ["like", "you know", "well", "I mean", "basically", "right"]
interjections_l1: ["哎", "唉", "哇", "嗯", "啊", "呀", "哦", "嘿", "诶"]
```

- [ ] **Step 2: Create `configs/zh_en/prompts.yaml`**

Extract from `prompt_generator.py:28-75` (CS templates), `prompt_generator.py:152-157` (proficiency), `sampling.py:292-301` (mixing levels), `dialogue_generator.py:346-368` (generation requirements + examples).

```yaml
role_template: "你是一个{persona}。{proficiency}"

cs_behavior_templates:
  ARC_01: |
    你的语言习惯：
    - 日常交流以{l1_name}为主，但会自然地嵌入{l2_name}单词和短语
    - {domain_context}
    - 常用的{l2_name}固定表达包括日常的 "by the way"、"make sense"、"no problem" 等
    - 切换通常发生在名词或名词短语的位置，你很少切换整句{l2_name}
    - 连续{l2_name}不超过3个词，绝不说完整{l2_name}句子
    - 句子的语法结构始终是{l1_name}的，{l2_name}词汇嵌入在{l1_name}句法框架中
  ARC_02: |
    你的语言习惯：
    - 你会根据话题和语境在{l1_name}{l2_name}之间自然切换
    - {domain_context}
    - 切换通常发生在句子之间或从句边界，每种语言内部的语法都是完整的
    - 你不会在一个短语中间突然换语言，而是在一个完整的思路单元结束后切换
    - 有时候引用别人的话或转述某个概念时，会保持原来的语言
  ARC_03: |
    你的语言习惯：
    - {l1_name}{l2_name}融合是你的自然说话方式，你在同一句话里会自由地在{l1_name}{l2_name}之间切换
    - {domain_context}
    - 你的切换非常频繁，有时一句话里来回切换好几次
    - 名词、动词、形容词都可能用{l2_name}，不限于术语
    - 这不是因为词汇缺口，而是你的日常语言习惯，哪个词先想到就用哪个
  ARC_04: |
    你的语言习惯：
    - 你主要用{l1_name}交流，但会在特定时刻策略性地切换{l2_name}
    - {domain_context}
    - 你切换{l2_name}的目的包括：强调某个观点、引用别人原话、开玩笑、表达正式立场
    - 你的切换不是随机的，每次用{l2_name}都有明确的交际目的
    - 有时你会用{l2_name}来显得更专业或更权威，有时用{l2_name}来制造幽默效果
  ARC_05: |
    你的语言习惯：
    - 你几乎完全用{l1_name}说话
    - {l2_name}只出现在品牌名、专有名词、或实在找不到{l1_name}对应词的情况
    - {domain_context}
    - {l2_name}单词在你的话语中非常少见，一段话里最多一两个
    - 你不会用{l2_name}短语或句子，偶尔出现的{l2_name}都是单个词
  ARC_06: |
    你的语言习惯：
    - 你的CS方式会根据和谁说话而显著变化
    - {domain_context}
    - 你会自然地向对方的语言习惯靠拢
    - 如果对方{l1_name}{l2_name}混合很多，你也会增加{l2_name}；如果对方主要说{l1_name}，你会收敛{l2_name}
    - 这种调整是自然的社交适应，不是刻意模仿

proficiency_descriptions:
  高级: "你的{l2_name}非常流利，能自如地在专业和日常场景中使用"
  中等: "你的{l2_name}水平中等，能使用{l2_name}短语和简单句，词汇量还不错"
  初级: "你的{l2_name}有限，仅能使用少量常见{l2_name}词汇"
  接近母语: "你的{l2_name}接近母语水平，在{l1_name}{l2_name}之间切换毫无障碍"

mixing_level_descriptions:
  minimal: "几乎不使用{l2_name}，仅限无法翻译的专有名词。一段话中最多1个{l2_name}词"
  light: "偶尔嵌入{l2_name}单词或短语，以{l1_name}为绝对主体。大约每10个词里有1个{l2_name}词"
  moderate: "适度混合{l1_name}{l2_name}，在术语和特定话题上自然切换。大约每5-6个词里有1个{l2_name}词"
  heavy: "频繁在{l1_name}{l2_name}之间切换，包括句内和句间切换。大约每3-4个词里有1个{l2_name}词"
  dense: "{l1_name}{l2_name}深度融合，两种语言在句内密集交替。大约每2-3个词里有1个{l2_name}词"

generation_requirements: |
  - 必须{l1_name}{l2_name}混合说话，在句子中自然地嵌入{l2_name}单词或短语
  - 【重要】长度严格控制在2-3句话，总字数不超过80字（含{l2_name}和标点），宁短勿长
  - 可以包含犹豫词和自我修正
  - 不要反复提及同一个话题信息的名称

example_dialogues:
  - "嗯那个meeting开完了，整个人有点exhausted，想找个cafe坐一会。"
  - "你有没有试过那个app？我觉得UI design还不错，就是loading有点慢。"
  - "昨天那个seminar讲的topic蛮interesting的，不过slides太多了看得我头晕。"
  - "最近在追一部Netflix的剧，plot twist特别多，每集都很intense。"
  - "这个weekend打算去hiking，天气forecast说会放晴。"
  - "我那个paper的revision快due了，reviewer的comments还没全部address。"
  - "刚试了楼下新开的brunch店，menu选择挺多但portion有点小。"
```

- [ ] **Step 3: Create `configs/zh_en/personas.yaml`**

Extract persona_templates from current `background_pools.yaml`. This file should contain the full persona_templates list and situation_pools (topics, relationships, formality, domain words) for zh_en. Copy the existing content from `stage1_infrastructure/background_pools.yaml`.

- [ ] **Step 4: Create `configs/zh_en/calibration.json`**

Copy from `stage1_infrastructure/corpus_calibration.json`.

- [ ] **Step 5: Create `configs/zh_en/evaluation.yaml`**

```yaml
# Evaluation rule overrides for zh_en
filler_words: ["嗯", "那个", "就是", "然后", "啊", "哎", "对", "like", "you know", "well", "I mean"]
interjections: ["哎", "唉", "哇", "嗯", "啊", "呀", "哦", "嘿", "诶"]

l2_span_expectations:
  ARC_01: { typical_max: 3, desc: "通常 1-3 词" }
  ARC_02: { typical_min: 4, desc: "通常整句/整从句" }
  ARC_03: { typical_max: 5, desc: "1-5 词频繁" }

switch_type_expectations:
  ARC_01: { intra_min: 0.70, desc: "Insertional: 应以句内嵌入为主" }
  ARC_02: { inter_min: 0.60, desc: "Alternational: 应以句间交替为主" }
  ARC_03: { intra_min: 0.50, min_switch_density: 0.15, desc: "Dense: 高频句内切换" }
  ARC_04: { desc: "Pragmatic: 灵活，无固定约束" }
  ARC_05: { max_cmi: 0.15, desc: "Reluctant: CMI 应很低" }
  ARC_06: { desc: "Accommodation: 适应型，无固定约束" }
```

- [ ] **Step 6: Commit**

```bash
git add configs/zh_en/
git commit -m "feat: create configs/zh_en/ extracted from hardcoded values"
```

---

### Task 3: Parameterize dialogue_generator.py

**Files:**
- Modify: `stage1_generate/dialogue_generator.py`

Key changes:
1. `GenerationConfig` adds `lang_pair: str = "zh_en"` field (replaces `language_pair` tuple)
2. `DialogueGenerator.__init__()` loads `LanguagePairConfig.load(config.lang_pair)`
3. `SpeakerAgent` receives `lang_config` and reads templates from it
4. `build_turn_prompt()` reads examples and requirements from config
5. `_clean_output()` uses `lang_config.l1_unicode_ranges` for Chinese detection
6. CLI: replace `--language-pair zh en` with `--lang-pair zh_en`
7. Output metadata uses `lang_config.pair_id`

- [ ] **Step 1: Update imports and GenerationConfig**

Add at top:
```python
from language_config import LanguagePairConfig
```

Change GenerationConfig:
```python
@dataclass
class GenerationConfig:
    num_dialogues: int = 100
    turns_per_dialogue: int = 6
    lang_pair: str = "zh_en"          # replaces language_pair tuple
    # ... rest unchanged
```

- [ ] **Step 2: Update DialogueGenerator.__init__() to load lang_config**

In `DialogueGenerator.__init__()`, after loading other configs:
```python
self.lang_config = LanguagePairConfig.load(config.lang_pair)
```

Pass `self.lang_config` to `SpeakerAgent`, evaluator, sampler.

- [ ] **Step 3: Update SpeakerAgent to use lang_config**

`SpeakerAgent.__init__()` receives `lang_config`. `_build_system_prompt()` reads `cs_behavior_templates`, `proficiency_descriptions`, `mixing_level_descriptions`, `role_template` from `lang_config` instead of hardcoded imports. Format templates with `l1_name=lang_config.l1_name, l2_name=lang_config.l2_name`.

- [ ] **Step 4: Update build_turn_prompt() to use lang_config**

Read `generation_requirements` and `example_dialogues` from `lang_config`. Use `random.choice(lang_config.example_dialogues)` for rotating examples.

- [ ] **Step 5: Update _clean_output() to use lang_config unicode ranges**

Replace hardcoded `re.search(r'[\u4e00-\u9fff]', ...)` with a dynamic check using `lang_config.l1_unicode_ranges`. Since `_clean_output` is currently a `@staticmethod`, change it to a regular method that accesses `self.lang_config`.

- [ ] **Step 6: Update CLI args and main()**

Replace `--language-pair` with `--lang-pair`. Add `--list-lang-pairs`. Update `GenerationConfig` construction. Add `--configs-root` optional arg.

- [ ] **Step 7: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('stage1_generate/dialogue_generator.py').read()); print('OK')"
git add stage1_generate/dialogue_generator.py
git commit -m "refactor: parameterize dialogue_generator.py with LanguagePairConfig"
```

---

### Task 4: Parameterize sampling.py

**Files:**
- Modify: `stage1_infrastructure/sampling.py`

Key changes:
1. `ContextualSampler.__init__()` accepts optional `lang_config` parameter
2. When `lang_config` is provided, use its `personas`, `calibration` instead of loading from files
3. `compute_language_mode()` reads `mixing_level_descriptions` from `lang_config`

- [ ] **Step 1: Update ContextualSampler to accept lang_config**

```python
def __init__(self, config_dir=None, lang_config=None):
    if lang_config is not None:
        self.pools = {"persona_templates": lang_config.personas,
                      "situation_pools": lang_config.personas_situation_pools}
        self.calibration = lang_config.calibration
        # archetypes remain universal
        arc_dir = Path(__file__).parent if config_dir is None else Path(config_dir)
        self.archetypes_data = self._load_yaml(arc_dir / "archetypes.yaml")
        self.lang_config = lang_config
    else:
        # existing logic unchanged for backwards compatibility
        ...
```

- [ ] **Step 2: Update compute_language_mode()**

Replace hardcoded description strings with `lang_config.mixing_level_descriptions`. Format with `l1_name` and `l2_name`.

- [ ] **Step 3: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('stage1_infrastructure/sampling.py').read()); print('OK')"
git add stage1_infrastructure/sampling.py
git commit -m "refactor: parameterize sampling.py with LanguagePairConfig"
```

---

### Task 5: Parameterize evaluator_agents.py

**Files:**
- Modify: `stage1_infrastructure/evaluator_agents.py`

Key changes:
1. `TextAnalyzer.__init__()` accepts `lang_config`
2. `_classify_char()` uses `lang_config.l1_unicode_ranges` instead of `_ZH_RANGES`
3. Token lang labels use `lang_config.l1_code`/`l2_code` instead of hardcoded "zh"/"en"
4. `_SENTENCE_ENDINGS` from `lang_config.sentence_endings`
5. `RuleBasedEvaluatorPipeline.__init__()` accepts `lang_config` and passes to checkers
6. `FluencyChecker`, `NaturalnessChecker` read filler words from config
7. `ProfileConsistencyChecker` reads span expectations from config evaluation overrides
8. Add lexicon detection mode support (for Phase 3)

- [ ] **Step 1: Update TextAnalyzer**

```python
class TextAnalyzer:
    def __init__(self, lang_config=None):
        if lang_config:
            self._l1_ranges = lang_config.l1_unicode_ranges
            self._l1_code = lang_config.l1_code
            self._l2_code = lang_config.l2_code
            self._sentence_endings = set(lang_config.sentence_endings)
            self._detection_mode = lang_config.detection_mode
            self._l1_lexicon = lang_config.l1_lexicon
        else:
            # backwards compatible defaults (zh_en)
            self._l1_ranges = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]
            self._l1_code = "zh"
            self._l2_code = "en"
            self._sentence_endings = set("。！？.!?")
            self._detection_mode = "unicode"
            self._l1_lexicon = None
```

- [ ] **Step 2: Update _classify_char() to use config ranges**

```python
def _classify_char(self, ch):
    cp = ord(ch)
    for start, end in self._l1_ranges:
        if start <= cp <= end:
            return self._l1_code
    if ch.isascii() and ch.isalpha():
        return self._l2_code
    if ch.isdigit():
        return "num"
    return "punct"
```

- [ ] **Step 3: Update _tokenize() to use l1_code/l2_code**

Replace all hardcoded `"zh"` with `self._l1_code` and `"en"` with `self._l2_code`.

- [ ] **Step 4: Add lexicon detection mode to _tokenize()**

For `detection_mode == "lexicon"` (dual-Latin pairs), all characters are Latin. Tokenize by whitespace, then classify each word by lexicon lookup:

```python
def _tokenize_lexicon(self, text):
    """Tokenize for dual-Latin language pairs using lexicon lookup."""
    tokens = []
    pos = 0
    for word in re.findall(r"[a-zA-Zàâæçéèêëïîôœùûüÿñáíóúü]+|[0-9]+|[^\w\s]", text):
        if word[0].isdigit():
            tokens.append(TokenInfo(text=word, lang="num", position=pos))
        elif word[0].isalpha():
            lang = self._classify_word_lexicon(word)
            tokens.append(TokenInfo(text=word, lang=lang, position=pos))
        else:
            tokens.append(TokenInfo(text=word, lang="punct", position=pos))
        pos += 1
    return tokens

def _classify_word_lexicon(self, word):
    normalized = word.lower()
    if self._l1_lexicon and normalized in self._l1_lexicon:
        return self._l1_code
    return self._l2_code
```

- [ ] **Step 5: Update RuleBasedEvaluatorPipeline to accept lang_config**

Pass `lang_config` to `TextAnalyzer` and all checkers.

- [ ] **Step 6: Update filler words in NaturalnessChecker/FluencyChecker**

Read from `lang_config.filler_words_l1`, `lang_config.interjections_l1` instead of hardcoded sets.

- [ ] **Step 7: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('stage1_infrastructure/evaluator_agents.py').read()); print('OK')"
git add stage1_infrastructure/evaluator_agents.py
git commit -m "refactor: parameterize evaluator_agents.py with LanguagePairConfig"
```

---

### Task 6: Parameterize prompt_generator.py and topic_information.py

**Files:**
- Modify: `stage1_infrastructure/prompt_generator.py`
- Modify: `stage1_generate/topic_information.py`

- [ ] **Step 1: Update prompt_generator.py**

The hardcoded `CS_BEHAVIOR_TEMPLATE_MAP` at lines 27-75 is now loaded from config. Keep the dict in the file as a fallback default, but `dialogue_generator.py` should prefer config values. Similarly for `PROFICIENCY_DESCRIPTIONS`.

No breaking changes needed — the module continues to export these constants. `dialogue_generator.py`'s `SpeakerAgent` will read from `lang_config` first, falling back to these.

- [ ] **Step 2: Update topic_information.py format_for_prompt()**

Replace hardcoded Chinese text with parameterized versions. `format_for_prompt()` should accept optional `l1_name`/`l2_name`:

```python
def format_for_prompt(self, snippets, language="zh", l1_name="中文", l2_name="英文"):
    if not snippets:
        return ""
    header = f"以下是一些与当前话题相关的真实信息，你们可以在对话中自然地讨论这些内容："
    # ... existing formatting ...
    lines.append(
        "注意：\n"
        "- 只需自然地提及其中1个你感兴趣的点即可\n"
        "- 不要照搬原文的词汇，用你自己日常的说法来表达\n"
        "- 不要反复提及信息来源的名称或标题\n"
        "- 把信息当作你已经知道的背景知识"
    )
```

- [ ] **Step 3: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('stage1_infrastructure/prompt_generator.py').read()); print('OK')"
python -c "import ast; ast.parse(open('stage1_generate/topic_information.py').read()); print('OK')"
git add stage1_infrastructure/prompt_generator.py stage1_generate/topic_information.py
git commit -m "refactor: parameterize prompt_generator.py and topic_information.py"
```

---

### Task 7: Phase 1 validation — zh_en produces identical results

- [ ] **Step 1: Run with --lang-pair zh_en and verify output structure**

```bash
python dialogue_generator.py --lang-pair zh_en --num-dialogues 5 --output output/test_zh_en.jsonl
```

- [ ] **Step 2: Verify output JSON structure matches previous format**

Check that dialogue JSON contains all expected fields, turns have text with Chinese-English mixing, scores are reasonable.

- [ ] **Step 3: Commit validation**

```bash
git commit --allow-empty -m "milestone: Phase 1 complete — zh_en parameterized and validated"
```

---

## Phase 2: CJK-Family Language Pairs

### Task 8: Create configs/yue_en/ (Cantonese-English)

**Files:**
- Create: `configs/yue_en/language.yaml`
- Create: `configs/yue_en/prompts.yaml`
- Create: `configs/yue_en/personas.yaml`
- Create: `configs/yue_en/calibration.json`
- Create: `configs/yue_en/evaluation.yaml`

- [ ] **Step 1: Create language.yaml** — Same CJK unicode ranges as zh_en plus Cantonese marker chars. Cantonese filler words: 啦, 喇, 咩, 嘛, 噃, 嗱, 係咪, 即係.

- [ ] **Step 2: Create prompts.yaml** — Archetype templates written for Cantonese context. Instructions in Mandarin, example dialogues in Cantonese. Examples like: "嗯我今日去咗個meeting，個project manager講嘅嘢都幾make sense。"

- [ ] **Step 3: Create personas.yaml** — Hong Kong locals, Macau, Vancouver/SF/Sydney/London Cantonese communities. Professions: office workers, students, media professionals, food industry.

- [ ] **Step 4: Create calibration.json** — Estimated CMI distributions based on HK bilingual patterns. Typical CMI ranges per archetype adjusted for Cantonese-English.

- [ ] **Step 5: Create evaluation.yaml** — Cantonese-specific filler words and interjections.

- [ ] **Step 6: Commit**

```bash
git add configs/yue_en/
git commit -m "feat: add configs/yue_en/ for Cantonese-English"
```

---

### Task 9: Create configs/ja_en/ (Japanese-English)

**Files:**
- Create: `configs/ja_en/language.yaml`
- Create: `configs/ja_en/prompts.yaml`
- Create: `configs/ja_en/personas.yaml`
- Create: `configs/ja_en/calibration.json`
- Create: `configs/ja_en/evaluation.yaml`

- [ ] **Step 1: Create language.yaml** — Unicode ranges for Hiragana (3040-309F), Katakana (30A0-30FF), CJK Kanji. Filler words: えーと, あの, まあ, ちょっと, なんか.

- [ ] **Step 2: Create prompts.yaml** — Templates in Japanese. Examples like: "えーと、今日のmeetingちょっと長かったね、あのproposalのfeedbackまだ来てないし。"

- [ ] **Step 3: Create personas.yaml** — Tokyo business professionals, 帰国子女, IT engineers, university students.

- [ ] **Step 4: Create calibration.json** — Japanese-English CS tends to have lower CMI than zh_en. Katakana loanword exclusion list included.

- [ ] **Step 5: Create evaluation.yaml** — Japanese-specific filler words, interjections, sentence endings (。！？).

- [ ] **Step 6: Commit**

```bash
git add configs/ja_en/
git commit -m "feat: add configs/ja_en/ for Japanese-English"
```

---

### Task 10: Create configs/min_en/ (Hokkien-English)

**Files:**
- Create: `configs/min_en/language.yaml`
- Create: `configs/min_en/prompts.yaml`
- Create: `configs/min_en/personas.yaml`
- Create: `configs/min_en/calibration.json`
- Create: `configs/min_en/evaluation.yaml`

- [ ] **Step 1-5:** Similar structure. CJK unicode ranges + Hokkien marker characters. Singapore/Malaysia/Taiwan Hokkien personas. Approximate calibration data (limited corpus). Note: set `max_retries_override: 8` as model may struggle with Hokkien generation.

- [ ] **Step 6: Commit**

```bash
git add configs/min_en/
git commit -m "feat: add configs/min_en/ for Hokkien-English"
```

---

### Task 11: Phase 2 milestone commit

- [ ] **Step 1: Commit milestone**

```bash
git commit --allow-empty -m "milestone: Phase 2 complete — CJK language pairs (yue_en, ja_en, min_en)"
```

---

## Phase 3: Latin-Script + Devanagari Language Pairs

### Task 12: Create configs/hi_en/ (Hindi-English)

**Files:**
- Create: `configs/hi_en/language.yaml`
- Create: `configs/hi_en/prompts.yaml`
- Create: `configs/hi_en/personas.yaml`
- Create: `configs/hi_en/calibration.json`
- Create: `configs/hi_en/evaluation.yaml`

- [ ] **Step 1: Create language.yaml** — Unicode Devanagari range (0900-097F). detection_mode: "unicode". Filler words: यार, अच्छा, बस, हाँ, ना, वो, मतलब.

- [ ] **Step 2: Create prompts.yaml** — Hindi instructions. Examples like: "यार आज का meeting बहुत long था, उस project की deadline भी approach कर रही है।"

- [ ] **Step 3: Create personas.yaml** — Indian urban youth (Delhi, Mumbai, Bangalore), IT professionals, university students.

- [ ] **Step 4: Create calibration.json** — Hinglish typically has high CMI (0.3-0.5). Based on SAIL/IIT datasets.

- [ ] **Step 5: Create evaluation.yaml** — Hindi filler words, Devanagari punctuation (। for full stop).

- [ ] **Step 6: Commit**

```bash
git add configs/hi_en/
git commit -m "feat: add configs/hi_en/ for Hindi-English (Hinglish)"
```

---

### Task 13: Create configs/fr_en/ (French-English)

**Files:**
- Create: `configs/fr_en/language.yaml`
- Create: `configs/fr_en/prompts.yaml`
- Create: `configs/fr_en/personas.yaml`
- Create: `configs/fr_en/calibration.json`
- Create: `configs/fr_en/evaluation.yaml`
- Create: `configs/fr_en/l1_lexicon.txt`

- [ ] **Step 1: Create language.yaml** — detection_mode: "lexicon". l1_script: "latin". French-specific Unicode for accented characters. Filler words: euh, ben, bah, genre, en fait, quoi, du coup.

- [ ] **Step 2: Create l1_lexicon.txt** — ~4000 French high-frequency words. Include function words (je, tu, il, elle, nous, vous, ils, de, du, des, le, la, les, un, une, et, ou, mais, donc, car, être, avoir, faire, aller, dire, voir, savoir, pouvoir, vouloir, etc.) and common content words.

- [ ] **Step 3: Create prompts.yaml** — French instructions. Examples like: "Euh, tu sais le meeting de ce matin était vraiment long, j'ai besoin d'un café là."

- [ ] **Step 4: Create personas.yaml** — Montreal bilinguals, Paris young professionals, Dakar/Abidjan francophone communities.

- [ ] **Step 5: Create calibration.json and evaluation.yaml**

- [ ] **Step 6: Commit**

```bash
git add configs/fr_en/
git commit -m "feat: add configs/fr_en/ for French-English"
```

---

### Task 14: Create configs/es_en/ (Spanish-English / Spanglish)

**Files:**
- Create: `configs/es_en/language.yaml`
- Create: `configs/es_en/prompts.yaml`
- Create: `configs/es_en/personas.yaml`
- Create: `configs/es_en/calibration.json`
- Create: `configs/es_en/evaluation.yaml`
- Create: `configs/es_en/l1_lexicon.txt`

- [ ] **Step 1: Create language.yaml** — detection_mode: "lexicon". Spanish punctuation includes ¿¡. Filler words: este, o sea, bueno, pues, verdad, mira, dale.

- [ ] **Step 2: Create l1_lexicon.txt** — ~4000 Spanish high-frequency words.

- [ ] **Step 3: Create prompts.yaml** — Spanglish examples like: "Oye, el meeting de hoy estuvo bien heavy, necesito un break antes del deadline."

- [ ] **Step 4: Create personas.yaml** — US Latinos (Chicano, Miami, NYC), Puerto Rico, Mexico City.

- [ ] **Step 5: Create calibration.json and evaluation.yaml** — Based on Miami Bangor corpus. Sentence endings include ¿¡.

- [ ] **Step 6: Commit**

```bash
git add configs/es_en/
git commit -m "feat: add configs/es_en/ for Spanish-English (Spanglish)"
```

---

### Task 15: Create configs/ms_en/ (Malay-English / Manglish)

**Files:**
- Create: `configs/ms_en/language.yaml`
- Create: `configs/ms_en/prompts.yaml`
- Create: `configs/ms_en/personas.yaml`
- Create: `configs/ms_en/calibration.json`
- Create: `configs/ms_en/evaluation.yaml`
- Create: `configs/ms_en/l1_lexicon.txt`

- [ ] **Step 1: Create language.yaml** — detection_mode: "lexicon". Malay discourse particles (lah, kan, meh) as filler words.

- [ ] **Step 2: Create l1_lexicon.txt** — ~3000 Malay high-frequency words + common affixed forms.

- [ ] **Step 3: Create prompts.yaml** — Manglish examples like: "Eh boss, hari ini punya meeting agak long lah, deadline tu macam impossible kan."

- [ ] **Step 4: Create personas.yaml** — Malaysian urban youth, Malaysian Chinese (bilingual context), Singaporean Malay speakers.

- [ ] **Step 5: Create calibration.json and evaluation.yaml**

- [ ] **Step 6: Commit**

```bash
git add configs/ms_en/
git commit -m "feat: add configs/ms_en/ for Malay-English (Manglish)"
```

---

### Task 16: Phase 3 milestone and final validation

- [ ] **Step 1: Run --list-lang-pairs to verify all 8 pairs are detected**

```bash
python dialogue_generator.py --list-lang-pairs
```

Expected output:
```
Available language pairs:
  es_en  (Español-English)
  fr_en  (Français-English)
  hi_en  (हिन्दी-English)
  ja_en  (日本語-English)
  min_en (闽南语-English)
  ms_en  (Bahasa Melayu-English)
  yue_en (粤語-English)
  zh_en  (中文-English)
```

- [ ] **Step 2: Generate test samples for each pair**

```bash
for pair in zh_en yue_en ja_en min_en hi_en fr_en es_en ms_en; do
  python dialogue_generator.py --lang-pair $pair --num-dialogues 3 --output output/test_${pair}.jsonl
done
```

- [ ] **Step 3: Final commit**

```bash
git commit --allow-empty -m "milestone: Phase 3 complete — all 8 language pairs implemented"
git push
```
