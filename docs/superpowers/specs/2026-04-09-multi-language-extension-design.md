# SwitchLingua 2.0 — Multi-Language Pair Extension Design

**Date:** 2026-04-09  
**Status:** Approved  
**Author:** Xielton + Claude

## 1. Overview

Extend SwitchLingua 2.0 from supporting only Mandarin-English code-switching to 8 language pairs. Each language pair is a self-contained configuration folder; core generation/evaluation code is shared and parameterized.

### Supported Language Pairs

| ID | L1 | L2 | Script Detection | Phase |
|----|----|----|-----------------|-------|
| `zh_en` | 普通话 (Mandarin) | English | Unicode (CJK) | 1 |
| `yue_en` | 粤语 (Cantonese) | English | Unicode (CJK + cantonese markers) | 2 |
| `ja_en` | 日本語 (Japanese) | English | Unicode (Kana + Kanji) | 2 |
| `min_en` | 闽南语 (Hokkien) | English | Unicode (CJK + hokkien markers) | 2 |
| `hi_en` | हिन्दी (Hindi) | English | Unicode (Devanagari) | 3 |
| `fr_en` | Français (French) | English | Lexicon | 3 |
| `es_en` | Español (Spanish) | English | Lexicon | 3 |
| `ms_en` | Bahasa Melayu (Malay) | English | Lexicon | 3 |

## 2. Architecture

### 2.1 Approach: Shared Engine + Language Config Folders

```
configs/
├── zh_en/          # One folder per language pair
│   ├── language.yaml       # Metadata, unicode ranges, filler words
│   ├── prompts.yaml        # Archetype templates, examples, instructions
│   ├── personas.yaml       # Persona pool for this language pair
│   ├── calibration.json    # CMI calibration data
│   └── evaluation.yaml     # Evaluation rule overrides
├── yue_en/
├── fr_en/
│   ├── ...
│   └── l1_lexicon.txt      # Lexicon mode only (dual-Latin pairs)
├── es_en/
├── hi_en/
├── ms_en/
├── ja_en/
└── min_en/
```

Core code (`stage1_generate/`, `stage1_infrastructure/`) is parameterized via `LanguagePairConfig`. CLI selects language pair with `--lang-pair zh_en`.

### 2.2 LanguagePairConfig

New file: `stage1_infrastructure/language_config.py`

```python
@dataclass
class LanguagePairConfig:
    pair_id: str               # "zh_en"
    l1_code: str               # "zh"
    l2_code: str               # "en"
    l1_name: str               # "中文"
    l2_name: str               # "英文"
    l1_name_en: str            # "Chinese"
    l2_name_en: str            # "English"
    l1_script: str             # "cjk" | "latin" | "devanagari" | "kana_kanji"
    l2_script: str             # "latin"
    detection_mode: str        # "unicode" | "lexicon"
    l1_unicode_ranges: list    # [[0x4E00, 0x9FFF], ...]
    l2_unicode_ranges: list
    sentence_endings: str
    l1_punctuation: str
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
    # Personas
    personas: list[dict]
    # Calibration
    calibration: dict
    # Evaluation overrides
    evaluation: dict
    # Lexicon (for dual-Latin pairs)
    l1_lexicon: set[str] | None = None
```

Loaded by `LanguagePairConfig.load(pair_id)` which reads all files in `configs/{pair_id}/`.

## 3. Configuration Details Per Language Pair

### 3.1 zh_en (普通话-英语)

Extracted from current hardcoded values. Serves as reference implementation.

- **Detection:** Unicode CJK `[\u4e00-\u9fff]`
- **Filler words L1:** 嗯, 那个, 就是, 然后, 啊, 哎
- **Filler words L2:** like, you know, well, I mean, basically
- **Personas:** Mainland China (researchers, students, tech workers), Singapore, Malaysia, Taiwan, Hong Kong, Overseas Chinese

### 3.2 yue_en (粤语-英语)

- **Detection:** Unicode CJK (same ranges as zh_en) + Cantonese-specific character markers
- **Cantonese-specific chars:** 嘅, 喺, 唔, 嗰, 咗, 嘢, 嚟, 乜, 冇, 噃, 嗱, 攞, 揸, 嘥
- **Grammar differences from Mandarin:** Different word order ("食咗饭未？"), different aspect markers (咗=了, 緊=正在, 晒=all), different possessive (嘅=的)
- **Filler words L1:** 啦, 喇, 咩, 嘛, 噃, 嗱, 係咪, 即係, 其實
- **Personas:** Hong Kong locals, Macau, overseas Cantonese communities (Vancouver, San Francisco, Sydney, London)
- **Prompt language:** Instructions in Mandarin Chinese, example dialogues in Cantonese
- **Calibration:** Based on HK Cantonese-English bilingual corpus patterns

### 3.3 ja_en (日本語-英語)

- **Detection:** Unicode Hiragana `[\u3040-\u309f]`, Katakana `[\u30a0-\u30ff]`, CJK Kanji
- **Key challenge:** Katakana loanwords (カタカナ外来語) vs genuine CS. Words like コンピューター (computer) are integrated loanwords, not CS. Config includes a katakana_loanword_exclude list.
- **Filler words L1:** えーと, あの, まあ, ちょっと, なんか, やっぱり
- **Personas:** Tokyo business professionals, returnee students (帰国子女), IT engineers, university students
- **Sentence endings:** 。！？.!?

### 3.4 min_en (闽南语-英语)

- **Detection:** Unicode CJK + Hokkien-specific character markers
- **Hokkien-specific chars/patterns:** 𪜶 (in), 予 (hō͘), 佇 (tī), 毋 (m̄), 閣 (koh)
- **Limited corpus data:** Calibration will use estimated ranges based on Singapore/Malaysia Hokkien community observations
- **Personas:** Singapore Hokkien speakers, Malaysian Hokkien speakers, Taiwan Hokkien (台語) speakers
- **Note:** Smallest research base among the 8 pairs; calibration data is approximate

### 3.5 hi_en (हिन्दी-English / Hinglish)

- **Detection:** Unicode Devanagari `[\u0900-\u097f]`
- **Script decision:** Use Devanagari (not Romanized Hindi) for clear detection
- **Filler words L1:** यार (yaar), अच्छा (accha), बस (bas), हाँ (haan), ना (na), वो (vo), मतलब (matlab)
- **Personas:** Indian urban youth, IT professionals, university students, Delhi/Mumbai/Bangalore
- **CS characteristics:** Hinglish is extremely common in daily life. Frequent intra-sentential switching. English verbs often take Hindi morphology (e.g., "download करो")
- **Calibration:** Based on SAIL/IIT Hinglish datasets

### 3.6 fr_en (Français-English)

- **Detection:** Lexicon mode (both Latin script)
- **L1 distinguishing features:** Accented chars (àâæçéèêëïîôœùûüÿ), French high-frequency words
- **Lexicon size:** ~4000 French high-frequency words
- **Ambiguous words handling:** Words shared between French and English (table, place, simple, question, important) marked as "ambiguous", excluded from CMI calculation
- **Filler words L1:** euh, ben, bah, genre, en fait, quoi, du coup, bref, voilà
- **Personas:** Montreal bilinguals, Paris young professionals, West African francophone communities (Dakar, Abidjan)
- **CS characteristics:** French grammar base + English noun insertion; or full clause alternation; gender agreement issues (le/la meeting?)

### 3.7 es_en (Español-English / Spanglish)

- **Detection:** Lexicon mode
- **L1 distinguishing features:** ñ, accents (áéíóúü), inverted punctuation (¿¡)
- **Lexicon size:** ~4000 Spanish high-frequency words
- **Filler words L1:** este, o sea, bueno, pues, ¿no?, verdad, mira, dale
- **Personas:** US Latinos (Chicano, Miami, NYC), Puerto Rico, Mexico City young professionals
- **CS characteristics:** Spanglish is the most studied CS variety. Frequent intra-sentential switching. English verbs may take Spanish morphology ("parquear" = to park). Well-documented community norms.
- **Calibration:** Based on Miami Bangor corpus, Spanglish NLP datasets
- **Sentence endings:** Include inverted marks in sentence boundary detection

### 3.8 ms_en (Bahasa Melayu-English / Manglish)

- **Detection:** Lexicon mode
- **L1 distinguishing features:** Malay affixes (-kan, -an, -nya, -lah, -kah), Malay high-frequency words
- **Lexicon size:** ~3000 Malay high-frequency words
- **Filler words L1:** lah, kan, eh, macam, apa, tu, ni
- **Personas:** Malaysian urban youth, Malaysian Chinese (trilingual context, but limited to ms-en pair), Singaporean Malay speakers
- **CS characteristics:** "Manglish" is a well-known variety. Discourse particles (lah, kan, meh) are frequently used even in English-dominant speech. Often three-way mixing in reality but we constrain to bilingual.

## 4. Code Parameterization

### 4.1 Modules to Modify

| Module | Changes |
|--------|---------|
| `stage1_infrastructure/language_config.py` | **NEW** — LanguagePairConfig dataclass + loader |
| `stage1_generate/dialogue_generator.py` | `GenerationConfig` adds `lang_config`; `SpeakerAgent` reads templates from config; `_clean_output()` uses config unicode ranges; CLI adds `--lang-pair` |
| `stage1_infrastructure/sampling.py` | `compute_language_mode()` reads descriptions from config; persona pool from config |
| `stage1_infrastructure/prompt_generator.py` | `CS_BEHAVIOR_TEMPLATE_MAP` loaded from config; `PROFICIENCY_DESCRIPTIONS` from config |
| `stage1_infrastructure/evaluator_agents.py` | `TextAnalyzer._classify_char()` uses config unicode ranges or lexicon; filler words from config; sentence boundaries from config; adds lexicon detection mode |
| `stage1_generate/topic_information.py` | `format_for_prompt()` uses `l1_name`/`l2_name` from config |

### 4.2 TextAnalyzer Detection Modes

**Unicode mode** (zh_en, yue_en, ja_en, min_en, hi_en):
```python
def _classify_char(self, char):
    code = ord(char)
    for lo, hi in self.config.l1_unicode_ranges:
        if lo <= code <= hi:
            return self.config.l1_code
    for lo, hi in self.config.l2_unicode_ranges:
        if lo <= code <= hi:
            return self.config.l2_code
    # punctuation, number, other...
```

**Lexicon mode** (fr_en, es_en, ms_en):
```python
def _classify_word(self, word):
    normalized = word.lower().strip(".,!?;:'\"")
    if normalized in self.config.l1_lexicon:
        if normalized in self._common_english:
            return "ambiguous"
        return self.config.l1_code
    return self.config.l2_code  # default to English
```

### 4.3 Unchanged Components

These remain language-agnostic:
- `archetypes.yaml` — 6 archetype definitions (Poplack/Gumperz theory is universal)
- `evaluation_constitutions.yaml` — Weights and thresholds
- CMI formula — Language-universal
- Accommodation controller — Giles CAT theory is universal
- Topic Router — Topic info fetching logic
- Diversity checker — n-gram novelty is language-agnostic

## 5. Implementation Phases

### Phase 1: Infrastructure + zh_en Extraction
1. Create `language_config.py` with `LanguagePairConfig`
2. Create `configs/zh_en/` (extract from current hardcoded values)
3. Parameterize all 5 .py modules
4. **Validation:** `--lang-pair zh_en` produces identical results to current code

### Phase 2: CJK-family Language Pairs
5. Create `configs/yue_en/`
6. Create `configs/ja_en/`
7. Create `configs/min_en/`
8. **Validation:** Generate sample dialogues for each, manual quality check

### Phase 3: Latin-script + Devanagari Language Pairs
9. Implement lexicon detection mode in `TextAnalyzer`
10. Create `configs/fr_en/` + `l1_lexicon.txt`
11. Create `configs/es_en/` + `l1_lexicon.txt`
12. Create `configs/ms_en/` + `l1_lexicon.txt`
13. Create `configs/hi_en/`
14. **Validation:** Generate sample dialogues for each, verify language detection accuracy

## 6. CLI Usage After Implementation

```bash
# Generate Mandarin-English CS data (default)
python dialogue_generator.py --lang-pair zh_en --num-dialogues 1000

# Generate Cantonese-English CS data
python dialogue_generator.py --lang-pair yue_en --num-dialogues 1000

# Generate Spanglish data
python dialogue_generator.py --lang-pair es_en --num-dialogues 1000

# List available language pairs
python dialogue_generator.py --list-lang-pairs
```

## 7. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Dual-Latin language detection accuracy | Lexicon approach + ambiguous word exclusion; validated against known CS corpora |
| Katakana loanwords conflated with CS | Exclude list of ~500 common katakana loanwords in ja_en config |
| Cantonese/Hokkien indistinguishable from Mandarin at character level | Use language-specific marker characters + prompt-level enforcement |
| Limited calibration data for min_en | Use estimated ranges; mark as approximate in config |
| Model (Qwen3.5) may not generate good Cantonese/Hindi text | Depends on model's training data; may need model-specific tuning or prompt engineering per language |
| Increased retry rates for less-common languages | Per-language max_retries configurable in language.yaml |
