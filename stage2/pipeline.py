"""
SwitchLingua 2.0 — Stage 2: Speech Synthesis Pipeline (Fish Speech S2)

Usage:
    python stage2/pipeline.py \
        --input output/zh_en_sample.jsonl \
        --output output/stage2/zh_en/ \
        --fish-url http://localhost:8080 \
        --profiles stage2/voice_profiles/profiles.yaml
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

from voice_assigner import VoiceAssigner
from tts_synthesizer import FishSpeechSynthesizer
from audio_assembler import AudioAssembler

logger = logging.getLogger("stage2_pipeline")


def process_dialogue(dialogue, voice_assigner, synthesizer, assembler, output_dir, profiles_dir):
    dlg_id = dialogue["dialogue_id"]
    dlg_dir = Path(output_dir) / dlg_id
    dlg_dir.mkdir(parents=True, exist_ok=True)

    try:
        voice_a, voice_b = voice_assigner.assign_voices(dialogue)
    except Exception as e:
        logger.error(f"[{dlg_id}] Voice assignment failed: {e}")
        return None

    voice_map = {"A": voice_a, "B": voice_b}
    logger.info(f"[{dlg_id}] Voices: A={voice_a.id}, B={voice_b.id}")

    turn_results = []
    turn_audio_files = []

    for turn in dialogue["turns"]:
        speaker = turn["speaker"]
        voice = voice_map[speaker]
        ref_audio = str(Path(profiles_dir) / voice.audio_file)
        if not Path(ref_audio).exists():
            logger.error(f"[{dlg_id}] Reference audio not found: {ref_audio}")
            return None

        try:
            result = synthesizer.synthesize_turn(
                text=turn["text"],
                reference_audio_path=ref_audio,
                output_dir=str(dlg_dir),
                turn_num=turn["turn"],
                speaker_name=speaker,
                reference_text=voice.transcript,
            )
            turn_results.append({
                "turn": turn["turn"],
                "speaker": speaker,
                "text": turn["text"],
                "audio_file": result["audio_file"],
                "duration_sec": result["duration_sec"],
            })
            turn_audio_files.append(result["audio_file"])
            logger.info(f"  Turn {turn['turn']} ({speaker}): {result['duration_sec']:.1f}s -> {result['audio_file']}")
        except Exception as e:
            logger.error(f"[{dlg_id}] Turn {turn['turn']} synthesis failed: {e}")
            return None

    try:
        assembly = assembler.assemble_dialogue(str(dlg_dir), turn_audio_files)
    except Exception as e:
        logger.error(f"[{dlg_id}] Audio assembly failed: {e}")
        assembly = {"full_audio": None, "full_duration_sec": 0}

    metadata = {
        "dialogue_id": dlg_id,
        "source_file": dialogue.get("_source_file", ""),
        "topic": dialogue.get("topic", ""),
        "relationship": dialogue.get("relationship", ""),
        "language_pair": dialogue.get("language_pair", []),
        "speaker_a_voice": voice_a.id,
        "speaker_b_voice": voice_b.id,
        "turns": turn_results,
        "full_audio": assembly.get("full_audio"),
        "full_duration_sec": assembly.get("full_duration_sec", 0),
    }

    with open(dlg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def run_pipeline(args):
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = str(Path(args.profiles).parent)

    voice_assigner = VoiceAssigner(args.profiles)
    synthesizer = FishSpeechSynthesizer(base_url=args.fish_url, timeout=args.timeout)
    assembler = AudioAssembler(pause_range=(args.pause_min, args.pause_max))

    if not synthesizer.check_health():
        logger.error(
            f"Fish Speech server not responding at {args.fish_url}. "
            f"Start with: bash stage2/deploy/launch_fish_speech.sh"
        )
        return

    dialogues = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dlg = json.loads(line)
                    dlg["_source_file"] = input_path.name
                    dialogues.append(dlg)
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(dialogues)} dialogues from {input_path}")
    if args.limit:
        dialogues = dialogues[:args.limit]
        logger.info(f"Limited to {len(dialogues)} dialogues")

    manifest = []
    success = failed = 0
    start_time = time.time()

    for i, dlg in enumerate(dialogues):
        dlg_id = dlg["dialogue_id"]
        logger.info(f"\n{'='*60}\n  [{i+1}/{len(dialogues)}] {dlg_id}\n{'='*60}")

        result = process_dialogue(dlg, voice_assigner, synthesizer, assembler, str(output_dir), profiles_dir)
        if result:
            manifest.append({
                "dialogue_id": dlg_id, "dir": dlg_id,
                "num_turns": len(result["turns"]),
                "full_duration_sec": result["full_duration_sec"],
                "speaker_a_voice": result["speaker_a_voice"],
                "speaker_b_voice": result["speaker_b_voice"],
            })
            success += 1
        else:
            failed += 1

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": str(input_path), "total_dialogues": len(dialogues),
            "success": success, "failed": failed,
            "elapsed_sec": round(time.time() - start_time, 1),
            "tts_engine": "fish-speech-s2", "dialogues": manifest,
        }, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n{'='*60}\n  Pipeline complete!\n"
        f"  Success: {success}/{len(dialogues)}\n  Failed: {failed}/{len(dialogues)}\n"
        f"  Time: {time.time()-start_time:.1f}s\n  Output: {output_dir}\n{'='*60}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Stage 2: Speech Synthesis (Fish Speech S2)")
    parser.add_argument("--input", required=True, help="Stage 1 JSONL file")
    parser.add_argument("--output", default="output/stage2/")
    parser.add_argument("--fish-url", default="http://localhost:8080", help="Fish Speech API URL")
    parser.add_argument("--profiles", default="stage2/voice_profiles/profiles.yaml")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--pause-min", type=int, default=300)
    parser.add_argument("--pause-max", type=int, default=800)
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    run_pipeline(args)


if __name__ == "__main__":
    main()
