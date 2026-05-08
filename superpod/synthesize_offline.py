"""
SwitchLingua 2.0 — Offline Speech Synthesis (No API Server)

直接加载 Fish Speech 模型到 GPU，本地完成全部合成，无需部署 API 服务。
适用于 SLURM/PBS 等任务调度集群。

Usage:
    python superpod/synthesize_offline.py \
        --input output/zh_en_dialogues.jsonl \
        --output output/stage2/zh_en/ \
        --asset-dir superpod/asset \
        --model-dir checkpoints/fish-speech-1.5 \
        --device cuda:0 \
        --limit 100
"""

import io
import json
import wave
import time
import struct
import random
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("offline_synth")


# ============================================================
# Reference Audio Manager
# ============================================================

class RefAudioManager:
    """Manages reference audio files: {lang}_{gender}_{id}.wav + .txt"""

    def __init__(self, asset_dir: str):
        self.asset_dir = Path(asset_dir)
        if not self.asset_dir.exists():
            raise FileNotFoundError(f"Asset directory not found: {asset_dir}")
        self._index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self._scan()

    def _scan(self):
        audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
        for f in sorted(self.asset_dir.iterdir()):
            if f.suffix.lower() not in audio_exts:
                continue
            parts = f.stem.split("_")
            if not parts:
                continue
            lang_code = parts[0].lower()
            gender = "unknown"
            if len(parts) >= 2 and parts[1].lower() in ("male", "female"):
                gender = parts[1].lower()

            txt_path = f.with_suffix(".txt")
            transcript = ""
            if txt_path.exists():
                transcript = txt_path.read_text(encoding="utf-8").strip()
            else:
                logger.warning(f"No transcript for {f.name}")

            self._index[lang_code].append((str(f), transcript, gender))

        for lang, refs in sorted(self._index.items()):
            logger.info(f"  {lang}: {len(refs)} ref(s)")

    def get_ref(self, lang_code: str, gender: str = "unknown") -> tuple[str, str]:
        lang_code = lang_code.lower()
        gender = gender.lower()
        refs = self._index.get(lang_code, [])
        if refs:
            if gender != "unknown":
                matched = [r for r in refs if r[2] == gender]
                if matched:
                    c = random.choice(matched)
                    return c[0], c[1]
            c = random.choice(refs)
            return c[0], c[1]
        if "en" in self._index:
            logger.warning(f"No ref for '{lang_code}', fallback to 'en'")
            c = random.choice(self._index["en"])
            return c[0], c[1]
        all_refs = [r for refs in self._index.values() for r in refs]
        if all_refs:
            c = random.choice(all_refs)
            return c[0], c[1]
        raise RuntimeError(f"No reference audio available")

    @property
    def available_languages(self) -> list[str]:
        return sorted(self._index.keys())


# ============================================================
# Fish Speech Offline Engine
# ============================================================

class FishSpeechOffline:
    """
    Load Fish Speech model directly and synthesize without API server.
    Supports Fish Speech 1.4/1.5.
    """

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load Fish Speech VQGAN + LLM models."""
        logger.info(f"Loading Fish Speech from {self.model_dir} on {self.device}...")
        t0 = time.time()

        try:
            # Fish Speech 1.5 loading path
            from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
            from fish_speech.utils.file import FISH_SPEECH_DIR
        except ImportError:
            raise ImportError(
                "fish-speech not installed. Run:\n"
                "  pip install fish-speech\n"
                "or clone and install:\n"
                "  git clone https://github.com/fishaudio/fish-speech.git\n"
                "  cd fish-speech && pip install -e ."
            )

        try:
            from tools.inference_engine import TTSInferenceEngine
            self.engine = TTSInferenceEngine(
                llama_checkpoint_path=str(self.model_dir),
                decoder_checkpoint_path=str(
                    self.model_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
                ),
                device=self.device,
                compile=False,
            )
            self._mode = "engine"
            logger.info(f"Loaded TTSInferenceEngine in {time.time()-t0:.1f}s")
        except (ImportError, Exception) as e:
            logger.info(f"TTSInferenceEngine not available ({e}), trying direct API...")
            # Fallback: use the lower-level API
            from tools.llama.generate import launch_thread_safe_queue
            from tools.vqgan.inference import load_model as load_decoder

            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(self.model_dir),
                device=self.device,
                compile=False,
                precision=torch.bfloat16,
            )
            self.decoder_model = load_decoder(
                config_name="firefly_gan_vq",
                checkpoint_path=str(
                    self.model_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
                ),
                device=self.device,
            )
            self._mode = "queue"
            logger.info(f"Loaded via queue mode in {time.time()-t0:.1f}s")

    def synthesize(self, text: str, ref_audio_path: str,
                   ref_text: str = "") -> bytes:
        """
        Synthesize speech from text using reference audio for voice cloning.

        Returns: WAV bytes
        """
        if self._mode == "engine":
            return self._synthesize_engine(text, ref_audio_path, ref_text)
        else:
            return self._synthesize_queue(text, ref_audio_path, ref_text)

    def _synthesize_engine(self, text: str, ref_audio_path: str,
                           ref_text: str) -> bytes:
        """Synthesize using TTSInferenceEngine."""
        from tools.inference_engine import ServeReferenceAudio, ServeTTSRequest

        ref_audio_bytes = Path(ref_audio_path).read_bytes()
        request = ServeTTSRequest(
            text=text,
            references=[
                ServeReferenceAudio(
                    audio=ref_audio_bytes,
                    text=ref_text or "reference audio",
                )
            ],
            format="wav",
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
        )

        result = self.engine.inference(request)

        # Collect audio chunks
        audio_chunks = []
        for chunk in result:
            if hasattr(chunk, 'audio') and chunk.audio:
                audio_chunks.append(chunk.audio)

        if not audio_chunks:
            raise RuntimeError("No audio generated")

        return b"".join(audio_chunks)

    def _synthesize_queue(self, text: str, ref_audio_path: str,
                          ref_text: str) -> bytes:
        """Synthesize using queue-based approach."""
        import soundfile as sf
        from tools.llama.generate import GenerateRequest, GenerateResponse
        from tools.vqgan.inference import decode as vqgan_decode

        # Encode reference audio
        ref_audio, ref_sr = sf.read(ref_audio_path)
        if len(ref_audio.shape) > 1:
            ref_audio = ref_audio.mean(axis=1)

        # Create generation request
        request = GenerateRequest(
            text=text,
            prompt_text=ref_text or "reference audio",
            prompt_tokens=None,  # Will be computed from ref audio
            references=[(ref_audio, ref_sr)],
        )

        # Generate tokens via LLM
        response: GenerateResponse = self.llama_queue.put(request)
        if response.codes is None:
            raise RuntimeError("LLM generation failed")

        # Decode tokens to audio via VQGAN
        audio_array = vqgan_decode(
            self.decoder_model,
            response.codes.to(self.device),
        )

        # Convert to WAV bytes
        audio_np = audio_array.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio_np, 44100, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ============================================================
# Audio Utils
# ============================================================

def wav_duration(wav_bytes: bytes) -> float:
    try:
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def concat_wavs(wav_list: list[bytes], pause_ms_range=(300, 800)) -> bytes:
    """Concatenate WAV files with random pauses between them."""
    if not wav_list:
        return b""

    import soundfile as sf

    all_audio = []
    target_sr = None

    for wav_bytes in wav_list:
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if target_sr is None:
            target_sr = sr
        all_audio.append(audio)

        # Add pause
        pause_ms = random.randint(*pause_ms_range)
        silence = np.zeros(int(target_sr * pause_ms / 1000))
        all_audio.append(silence)

    # Remove last silence
    if all_audio:
        all_audio.pop()

    full_audio = np.concatenate(all_audio)
    buf = io.BytesIO()
    sf.write(buf, full_audio, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ============================================================
# Gender Inference
# ============================================================

def infer_gender(persona_desc: str) -> str:
    desc = persona_desc.lower()
    female_kw = ["女", "female", "她", "mother", "sister", "妈", "姐", "妹",
                 "lady", "woman", "여성", "женщин", "perempuan"]
    male_kw = ["男", "male", "他", "father", "brother", "爸", "哥", "弟",
               "man", "남성", "мужчин", "lelaki"]
    f = sum(1 for k in female_kw if k in desc)
    m = sum(1 for k in male_kw if k in desc)
    if f > m: return "female"
    if m > f: return "male"
    return "unknown"


# ============================================================
# Main Pipeline
# ============================================================

def process_dialogue(dlg: dict, engine: FishSpeechOffline,
                     ref_manager: RefAudioManager, output_dir: Path) -> Optional[dict]:
    dlg_id = dlg["dialogue_id"]
    dlg_dir = output_dir / dlg_id
    dlg_dir.mkdir(parents=True, exist_ok=True)

    lang_pair = dlg.get("language_pair", [])
    l1 = lang_pair[0] if lang_pair else "en"

    # Assign different voices for A and B
    gender_a = infer_gender(dlg.get("speaker_a", {}).get("persona_description", ""))
    gender_b = infer_gender(dlg.get("speaker_b", {}).get("persona_description", ""))
    if gender_a == gender_b == "unknown":
        gender_a, gender_b = "male", "female"

    ref_a, ref_text_a = ref_manager.get_ref(l1, gender_a)
    ref_b, ref_text_b = ref_manager.get_ref(l1, gender_b)
    if ref_a == ref_b:
        opposite = "female" if gender_a == "male" else "male"
        ref_b, ref_text_b = ref_manager.get_ref(l1, opposite)

    voice_map = {"A": (ref_a, ref_text_a), "B": (ref_b, ref_text_b)}
    logger.info(f"  L1={l1}, A={Path(ref_a).name} ({gender_a}), B={Path(ref_b).name} ({gender_b})")

    turn_results = []
    turn_wavs = []

    for turn in dlg["turns"]:
        turn_num = turn["turn"]
        speaker = turn["speaker"]
        text = turn["text"]
        ref_audio, ref_text = voice_map[speaker]
        filename = f"turn_{turn_num}_{speaker}.wav"
        filepath = dlg_dir / filename

        try:
            logger.info(f"  Turn {turn_num} ({speaker}): {text[:50]}...")
            wav_bytes = engine.synthesize(text, ref_audio, ref_text)
            filepath.write_bytes(wav_bytes)
            dur = wav_duration(wav_bytes)

            turn_results.append({
                "turn": turn_num, "speaker": speaker, "text": text,
                "audio_file": filename, "duration_sec": round(dur, 2),
            })
            turn_wavs.append(wav_bytes)
            logger.info(f"    -> {filename} ({dur:.2f}s)")
        except Exception as e:
            logger.error(f"  Turn {turn_num} FAILED: {e}")
            turn_results.append({
                "turn": turn_num, "speaker": speaker, "text": text,
                "audio_file": None, "error": str(e),
            })

    # Assemble full dialogue
    full_audio_file = None
    full_duration = 0.0
    if turn_wavs:
        try:
            full_wav = concat_wavs(turn_wavs)
            (dlg_dir / "dialogue_full.wav").write_bytes(full_wav)
            full_duration = wav_duration(full_wav)
            full_audio_file = "dialogue_full.wav"
        except Exception as e:
            logger.error(f"  Assembly failed: {e}")

    metadata = {
        "dialogue_id": dlg_id, "language_pair": lang_pair,
        "topic": dlg.get("topic", ""), "relationship": dlg.get("relationship", ""),
        "voices": {
            "A": {"ref_audio": Path(ref_a).name, "gender": gender_a},
            "B": {"ref_audio": Path(ref_b).name, "gender": gender_b},
        },
        "turns": turn_results,
        "full_audio": full_audio_file,
        "full_duration_sec": round(full_duration, 2),
    }
    with open(dlg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 — Offline Speech Synthesis (No API Server)")
    parser.add_argument("--input", required=True, help="Stage 1 JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--asset-dir", required=True, help="Reference audio directory")
    parser.add_argument("--model-dir", required=True,
                        help="Fish Speech model checkpoint directory")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cuda:1, etc.)")
    parser.add_argument("--limit", type=int, default=0, help="Max dialogues (0=all)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Init reference audio manager
    ref_manager = RefAudioManager(args.asset_dir)
    logger.info(f"Available languages: {ref_manager.available_languages}")

    # Init Fish Speech engine (loads model to GPU)
    engine = FishSpeechOffline(args.model_dir, device=args.device)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dialogues
    dialogues = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dialogues.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if args.limit:
        dialogues = dialogues[:args.limit]
    logger.info(f"Loaded {len(dialogues)} dialogues")

    # Process
    manifest = []
    success = failed = 0
    t0 = time.time()

    for i, dlg in enumerate(dialogues):
        dlg_id = dlg["dialogue_id"]
        logger.info(f"\n{'='*60}\n  [{i+1}/{len(dialogues)}] {dlg_id}\n{'='*60}")

        result = process_dialogue(dlg, engine, ref_manager, output_dir)
        if result and result.get("full_audio"):
            manifest.append({
                "dialogue_id": dlg_id,
                "num_turns": len(result["turns"]),
                "full_duration_sec": result["full_duration_sec"],
                "voices": result.get("voices", {}),
            })
            success += 1
        else:
            failed += 1

    # Write manifest
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": args.input, "tts_engine": "fish-speech-offline",
            "model_dir": str(args.model_dir),
            "device": args.device,
            "total": len(dialogues), "success": success, "failed": failed,
            "elapsed_sec": round(time.time() - t0, 1),
            "dialogues": manifest,
        }, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info(
        f"\n{'='*60}\n  Done! {success}/{len(dialogues)} success, "
        f"{failed} failed, {elapsed:.1f}s ({elapsed/max(success,1):.1f}s/dlg)"
        f"\n  Output: {output_dir}\n{'='*60}"
    )


if __name__ == "__main__":
    main()
