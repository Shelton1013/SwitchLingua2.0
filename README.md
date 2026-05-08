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

```
SwitchLingua2.0/
├── stage1_infrastructure/     # Theoretical foundations & data structures
│   ├── archetypes.yaml        # 10 CS behavior archetypes
│   ├── background_pools.yaml  # Speaker background pools
│   ├── sampling.py            # Contextual parameter sampler
│   ├── prompt_generator.py    # Prompt template engine
│   ├── evaluator_agents.py    # Rule-based evaluator pipeline
│   └── language_config.py     # Language pair configuration
│
├── stage1_generate/           # Dialogue generation engine
│   ├── dialogue_generator.py  # Main generator with dual speaker agents
│   ├── topic_information.py   # Real-time topic info injection (MCP)
│   └── provider_config.yaml   # LLM provider configuration
│
├── stage1.5/                  # Post-generation LLM evaluation
│   ├── llm_evaluator.py       # 4+1 agent evaluation framework
│   ├── auto_metrics.py        # Automatic descriptive metrics (CMI, etc.)
│   └── analyze_results.py     # Evaluation result analysis
│
├── stage2/                    # Speech synthesis pipeline
│   ├── pipeline.py            # Main synthesis pipeline
│   ├── voice_assigner.py      # Voice profile matching
│   ├── tts_synthesizer.py     # TTS engine wrapper
│   ├── audio_assembler.py     # Multi-turn audio assembly
│   ├── batch_synthesize.py    # Batch processing script
│   └── voice_profiles/       # Voice profile definitions
│
├── configs/                   # Language-pair-specific configurations
│   └── {lang}_en/             # Prompts, personas, evaluation rules
│
├── stage2_offline/            # Offline TTS (direct model loading)
└── superpod/                  # HPC/SLURM deployment scripts
```

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
    --lang-pair zh_en \
    --output output/zh_en_dialogues.jsonl
```

### Stage 1.5: Quality Evaluation

```bash
# LLM-based evaluation
python stage1.5/llm_evaluator.py \
    --input output/zh_en_dialogues.jsonl \
    --output output/stage1.5/zh_en_eval.jsonl \
    --api-base http://localhost:8001/v1 \
    --model Qwen3.5-122B-A10B-FP8 \
    --lang-pair zh_en

```

### Stage 2: Speech Synthesis

```bash
# Option A: API-based (requires running Fish Speech / CosyVoice server)
python stage2/pipeline.py \
    --input output/zh_en_dialogues.jsonl \
    --output output/stage2/zh_en/ \
    --fish-url http://localhost:8080 \
    --profiles stage2/voice_profiles/profiles.yaml

# Option B: Offline (direct model loading, no server needed)
python stage2_offline/synthesize.py \
    --input output/zh_en_dialogues.jsonl \
    --output output/stage2/zh_en/ \
    --asset-dir stage2_offline/asset \
    --model-dir ~/models/s2-pro
```

## Baselines

Run all 8 baselines for comparison:

```bash
bash baselines/run_all.sh \
    --api-base http://localhost:8001/v1 \
    --model Qwen3.5-122B-A10B-FP8 \
    --lang-pair zh_en \
    --num 200
```

| # | Baseline | Reference |
|---|----------|-----------|
| 1 | Naive Prompting | — |
| 2 | Template-Based | Pratapa et al. (2018) |
| 3 | UniCoM/SWORDS | EMNLP 2025 |
| 4 | EZSwitch | Kuwanto et al. (2024) |
| 5 | Few-Shot Prompting | Yong et al. (2023) |
| 6 | MCE/MADGF | ICASSP 2025 |
| 7 | SwitchLingua 1.0 | NeurIPS 2025 |
| 8 | Persona-Only | Ablation |

## Adding a New Language Pair

1. Create a config directory: `configs/{lang}_en/`
2. Define the required YAML files (see `configs/zh_en/` as template):
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
