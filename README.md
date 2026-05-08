<p align="center">
  <img src="logo_git.png" alt="SwitchLingua V2" width="600"/>
</p>


<p align="center">
  <strong>SwitchLingua V2: Agent-Driven Code-Switching via Digital Clones</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#supported-languages">Languages</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#baselines">Baselines</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

SwitchLingua 2.0 is a comprehensive framework for generating high-quality, naturalistic **code-switching (CS) dialogue data** and synthesizing the corresponding **speech**. It addresses the scarcity of CS training data by combining linguistic theory, persona-driven generation, and multi-stage quality control.


## Supported Languages

All language pairs are `X ↔ English`:

| Language | Config | Language | Config |
|----------|--------|----------|--------|
| Chinese Mandarin (zh) | `zh_en` | Korean (ko) | `ko_en` |
| Cantonese (yue) | `yue_en` | Arabic (ar) | `ar_en` |
| Japanese (ja) | `ja_en` | Russian (ru) | `ru_en` |
| Malay (ms) | `ms_en` | Turkish (tr) | `tr_en` |
| Minangkabau (min) | `min_en` | Thai (th) | `th_en` |
| Hindi (hi) | `hi_en` | Italian (it) | `it_en` |
| Spanish (es) | `es_en` | French (fr) | `fr_en` |
| German (de) | `de_en` | | |

## Project Structure


## Quick Start

### Prerequisites

- Python 3.10+
- A running vLLM server (for dialogue generation)
- PyYAML, requests

### Stage 1: Generate CS Dialogues

```bash
# 1. Start a vLLM server with your preferred model
# 2. Run the generator
python stage1_generate/dialogue_generator.py \
    --num-dialogues 100 \
    --turns-per-dialogue 6 \
    --api-base http://localhost:8001/v1 \
    --model Qwen3.5-122B-A10B-FP8 \
    --lang-pair yue_en \
    --output output/yue_en_dialogues.jsonl
```

### Stage 2: Speech Synthesis

```bash
# Option A: API-based (requires running Fish Speech server)
python stage2/pipeline.py \
    --input output/zh_en_dialogues.jsonl \
    --output output/stage2/yue_en/ \
    --fish-url http://localhost:8080 \
    --profiles stage2/voice_profiles/profiles.yaml

# Option B: Offline (direct model loading, no server needed)
python stage2_offline/synthesize.py \
    --input output/zh_en_dialogues.jsonl \
    --output output/stage2/yue_en/ \
    --asset-dir stage2_offline/asset \
    --model-dir ~/models/s2-pro
```

## Adding a New Language Pair

1. Create a config directory: `configs/{lang}_en/`
2. Define the required YAML files (see `configs/yue_en/` as template):
   - `language.yaml` — language metadata and L1/L2 names
   - `prompts.yaml` — archetype-specific prompt templates
   - `personas.yaml` — speaker persona definitions
   - `evaluation.yaml` — language-specific evaluation rules
   - `calibration.json` — CMI calibration from reference corpus
3. Run the generator with `--lang-pair {lang}_en`

## Citation

```bibtex

```

## License

This project is released for research purposes. Please cite the paper if you use it in your work.
