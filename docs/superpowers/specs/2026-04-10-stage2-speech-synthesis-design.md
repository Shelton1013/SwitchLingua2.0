# SwitchLingua 2.0 — Stage 2: Speech Synthesis Pipeline Design

**Date:** 2026-04-10
**Status:** Approved

## 1. Overview

Stage 2 takes Stage 1's text dialogue JSONL output and synthesizes speech audio using CosyVoice 3 (zero-shot voice cloning TTS). Each speaker in a dialogue is assigned a voice profile based on their persona attributes, then each turn is synthesized individually and also assembled into a full dialogue audio.

## 2. Pipeline Flow

```
Stage 1 JSONL → pipeline.py → output/stage2/{dataset}/
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 voice_assigner  tts_synth   audio_assembler
 (persona→voice) (CosyVoice3) (concat+pause)
```

## 3. Input / Output

**Input:** `output/zh_en_sample.jsonl` (or any Stage 1 JSONL)

**Output:**
```
output/stage2/{dataset_name}/
├── {dialogue_id}/
│   ├── turn_1_A.wav
│   ├── turn_2_B.wav
│   ├── ...
│   ├── dialogue_full.wav      # concatenated with natural pauses
│   └── metadata.json          # per-dialogue metadata
└── manifest.json              # global index
```

## 4. Modules

### 4.1 voice_assigner.py
- Loads `voice_profiles/profiles.yaml`
- Maps persona attributes (language, gender, age, region) to voice profiles
- Ensures A and B in same dialogue get different voices
- Gender inference from persona_description keywords

### 4.2 tts_synthesizer.py
- Wraps CosyVoice 3 FastAPI `/inference_zero_shot` endpoint
- Input: text + reference audio path → output: wav bytes
- Retry on failure (max 3)
- Timeout: 60s per request
- Saves each turn as individual wav file

### 4.3 audio_assembler.py
- Reads per-turn wav files for one dialogue
- Inserts random pause (300-800ms silence) between turns
- Outputs `dialogue_full.wav`
- Uses `wave` stdlib module (no ffmpeg dependency)
- Normalizes sample rate to 16kHz mono

### 4.4 pipeline.py (CLI entry point)
```bash
python stage2/pipeline.py \
    --input output/zh_en_sample.jsonl \
    --output output/stage2/zh_en_sample/ \
    --cosyvoice-url http://localhost:50000 \
    --profiles stage2/voice_profiles/profiles.yaml
```

### 4.5 deploy/launch_cosyvoice.sh
CosyVoice 3 local deployment script for server with GPU.

## 5. Voice Profiles

`voice_profiles/profiles.yaml`:
```yaml
profiles:
  - id: "zh_male_young_01"
    gender: "male"
    age_group: "young"
    languages: ["zh", "en"]
    accent: "mainland"
    audio_file: "audio/zh_male_young_01.wav"
```

Initial set: ~10 profiles covering main gender/age/language combinations.
Reference audio: 3-10s clean speech clips from public datasets or recordings.

## 6. CosyVoice 3 Deployment

- Model: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Interface: FastAPI server on port 50000
- GPU: ~4-6GB VRAM
- Supported languages: zh, en, ja, ko, de, es, fr, it, ru + Cantonese dialects
- NOT supported: Hindi (hi), Malay (ms) — these pairs will need alternative TTS

## 7. Metadata Format

Per-dialogue `metadata.json`:
```json
{
  "dialogue_id": "DLG_abc123",
  "source_file": "zh_en_sample.jsonl",
  "speaker_a_voice": "zh_male_young_01",
  "speaker_b_voice": "zh_female_young_01",
  "turns": [
    {"turn": 1, "speaker": "A", "text": "...", "audio_file": "turn_1_A.wav", "duration_sec": 3.2},
    {"turn": 2, "speaker": "B", "text": "...", "audio_file": "turn_2_B.wav", "duration_sec": 4.1}
  ],
  "full_audio": "dialogue_full.wav",
  "full_duration_sec": 18.5
}
```
