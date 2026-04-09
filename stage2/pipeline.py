"""
SwitchLingua 2.0 — Stage 2: Speech Synthesis Pipeline

Reads Stage 1 dialogue JSONL, assigns voice profiles to speakers,
synthesizes each turn via CosyVoice 3, and assembles full dialogue audio.

Usage:
    python stage2/pipeline.py \
        --input output/zh_en_sample.jsonl \
        --output output/stage2/zh_en_sample/ \
        --cosyvoice-url http://localhost:50000 \
        --profiles stage2/voice_profiles/profiles.yaml
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

from voice_assigner import VoiceAssigner
from tts_synthesizer import CosyVoiceSynthesizer
from audio_assembler import AudioAssembler

logger = logging.getLogger("stage2_pipeline")


def process_dialogue(
    dialogue: dict,
    voice_assigner: VoiceAssigner,
    synthesizer: CosyVoiceSynthesizer,
    assembler: AudioAssembler,
    output_dir: str,
    profiles_dir: str,
) -> Optional[dict]:
    """Process one dialogue: assign voices → synthesize turns → assemble.

    Returns metadata dict on success, None on failure.
    """
    dlg_id = dialogue["dialogue_id"]
    dlg_dir = Path(output_dir) / dlg_id
    dlg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Assign voice profiles
    try:
        voice_a, voice_b = voice_assigner.assign_voices(dialogue)
    except Exception as e:
        logger.error(f"[{dlg_id}] Voice assignment failed: {e}")
        return None

    voice_map = {"A": voice_a, "B": voice_b}
    logger.info(
        f"[{dlg_id}] Voices: A={voice_a.id}, B={voice_b.id}"
    )

    # 2. Synthesize each turn
    turn_results = []
    turn_audio_files = []

    for turn in dialogue["turns"]:
        speaker = turn["speaker"]
        voice = voice_map[speaker]

        # Resolve reference audio path
        ref_audio = str(Path(profiles_dir) / voice.audio_file)
        if not Path(ref_audio).exists():
            logger.error(
                f"[{dlg_id}] Reference audio not found: {ref_audio}"
            )
            return None

        try:
            result = synthesizer.synthesize_turn(
                text=turn["text"],
                reference_audio_path=ref_audio,
                output_dir=str(dlg_dir),
                turn_num=turn["turn"],
                speaker_name=speaker,
            )
            turn_results.append(
                {
                    "turn": turn["turn"],
                    "speaker": speaker,
                    "text": turn["text"],
                    "audio_file": result["audio_file"],
                    "duration_sec": result["duration_sec"],
                }
            )
            turn_audio_files.append(result["audio_file"])
            logger.info(
                f"  Turn {turn['turn']} ({speaker}): "
                f"{result['duration_sec']:.1f}s → {result['audio_file']}"
            )
        except Exception as e:
            logger.error(
                f"[{dlg_id}] Turn {turn['turn']} synthesis failed: {e}"
            )
            return None

    # 3. Assemble full dialogue audio
    try:
        assembly = assembler.assemble_dialogue(str(dlg_dir), turn_audio_files)
    except Exception as e:
        logger.error(f"[{dlg_id}] Audio assembly failed: {e}")
        assembly = {"full_audio": None, "full_duration_sec": 0}

    # 4. Write metadata
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

    meta_path = dlg_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def run_pipeline(args):
    """Main pipeline entry point."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_dir = str(Path(args.profiles).parent)

    # Initialize components
    voice_assigner = VoiceAssigner(args.profiles)
    synthesizer = CosyVoiceSynthesizer(
        base_url=args.cosyvoice_url,
        timeout=args.timeout,
    )
    assembler = AudioAssembler(
        pause_range=(args.pause_min, args.pause_max),
    )

    # Health check
    if not synthesizer.check_health():
        logger.error(
            f"CosyVoice server not responding at {args.cosyvoice_url}. "
            f"Start it with: bash stage2/deploy/launch_cosyvoice.sh"
        )
        return

    # Load dialogues
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
        dialogues = dialogues[: args.limit]
        logger.info(f"Limited to {len(dialogues)} dialogues")

    # Process dialogues
    manifest = []
    success = 0
    failed = 0
    start_time = time.time()

    for i, dlg in enumerate(dialogues):
        dlg_id = dlg["dialogue_id"]
        logger.info(
            f"\n{'='*60}\n"
            f"  [{i+1}/{len(dialogues)}] {dlg_id}\n"
            f"{'='*60}"
        )

        result = process_dialogue(
            dlg, voice_assigner, synthesizer, assembler,
            str(output_dir), profiles_dir,
        )

        if result:
            manifest.append(
                {
                    "dialogue_id": dlg_id,
                    "dir": dlg_id,
                    "num_turns": len(result["turns"]),
                    "full_duration_sec": result["full_duration_sec"],
                    "speaker_a_voice": result["speaker_a_voice"],
                    "speaker_b_voice": result["speaker_b_voice"],
                }
            )
            success += 1
        else:
            failed += 1

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(input_path),
                "total_dialogues": len(dialogues),
                "success": success,
                "failed": failed,
                "elapsed_sec": round(time.time() - start_time, 1),
                "dialogues": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.time() - start_time
    logger.info(
        f"\n{'='*60}\n"
        f"  Pipeline complete!\n"
        f"  Success: {success}/{len(dialogues)}\n"
        f"  Failed:  {failed}/{len(dialogues)}\n"
        f"  Time:    {elapsed:.1f}s\n"
        f"  Output:  {output_dir}\n"
        f"{'='*60}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Stage 2: Speech Synthesis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to Stage 1 JSONL file",
    )
    parser.add_argument(
        "--output", default="output/stage2/",
        help="Output directory for synthesized audio",
    )
    parser.add_argument(
        "--cosyvoice-url", default="http://localhost:50000",
        help="CosyVoice 3 server URL",
    )
    parser.add_argument(
        "--profiles",
        default="stage2/voice_profiles/profiles.yaml",
        help="Path to voice profiles YAML",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="TTS request timeout in seconds",
    )
    parser.add_argument(
        "--pause-min", type=int, default=300,
        help="Minimum pause between turns (ms)",
    )
    parser.add_argument(
        "--pause-max", type=int, default=800,
        help="Maximum pause between turns (ms)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of dialogues to process (0 = all)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_pipeline(args)


if __name__ == "__main__":
    main()
