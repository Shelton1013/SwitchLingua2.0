"""
SwitchLingua 2.0 — Stage 2 Offline Synthesis (Direct Model Loading)

直接加载 Fish Speech 模型，一个脚本完成全部合成，无需部署 API 服务器。
支持 Fish Speech v2.0 (S2-Pro) 和 v1.5 两个版本。

与 stage2/batch_synthesize.py 的区别：
  - batch_synthesize.py: 通过 HTTP API 调用已部署的 Fish Speech 服务
  - 本脚本: 直接 import 模型到 GPU，单进程完成全部推理

适用场景：
  - SLURM/PBS 集群提交 GPU 任务
  - 本地单机 GPU 合成
  - 无法部署常驻服务的环境

Usage:
    python stage2_offline/synthesize.py \
        --input output/zh_en_dialogues.jsonl \
        --output output/stage2/zh_en/ \
        --asset-dir stage2_offline/asset \
        --model-dir checkpoints/s2-pro \
        --version v2 \
        --device cuda:0 \
        --limit 100

Asset 目录结构（与 stage2 相同）：
    asset/
    ├── zh_male_01.wav      # {lang}_{gender}_{id}.wav
    ├── zh_male_01.txt      # 对应的文字转录
    ├── zh_female_01.wav
    ├── zh_female_01.txt
    └── ...
"""

import io
import json
import wave
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage2_offline")


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
        raise RuntimeError("No reference audio available")

    @property
    def available_languages(self) -> list[str]:
        return sorted(self._index.keys())


# ============================================================
# Fish Speech Engine — v2.0 (S2-Pro)
# ============================================================

class FishSpeechV2:
    """
    Fish Speech v2.0 (S2-Pro) offline engine.
    Uses DualARTransformer + DAC codec.
    Requires ~24GB VRAM.
    """

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        import torch
        self.device = device
        self.model_dir = Path(model_dir)
        self.precision = torch.bfloat16

        logger.info(f"Loading Fish Speech v2.0 from {model_dir}...")
        t0 = time.time()

        from fish_speech.models.text2semantic.inference import (
            init_model,
            load_codec_model,
        )

        # Load LLM
        self.model, self.decode_one_token = init_model(
            self.model_dir, device, self.precision, compile=False
        )
        with torch.device(device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=self.precision,
            )

        # Load codec
        codec_path = self.model_dir / "codec.pth"
        self.codec = load_codec_model(codec_path, device, self.precision)

        logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    def synthesize(self, text: str, ref_audio_path: str,
                   ref_text: str = "") -> bytes:
        """Synthesize speech. Returns WAV bytes."""
        import torch
        import soundfile as sf
        from fish_speech.models.text2semantic.inference import (
            encode_audio,
            generate_long,
            decode_to_audio,
        )

        # Encode reference audio
        ref_codes = encode_audio(ref_audio_path, self.codec, self.device)

        # Generate
        tagged_text = f"<|speaker:0|>{text}"
        tagged_ref = f"<|speaker:0|>{ref_text}" if ref_text else "<|speaker:0|>reference"

        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=tagged_text,
            prompt_text=[tagged_ref],
            prompt_tokens=[ref_codes.cpu()],
            top_p=0.9,
            top_k=30,
            temperature=1.0,
            max_new_tokens=0,
            chunk_length=300,
        )

        all_codes = []
        for response in generator:
            if response.action == "sample":
                all_codes.append(response.codes)
            elif response.action == "next":
                break

        if not all_codes:
            raise RuntimeError("No audio generated")

        # Decode to audio
        merged_codes = torch.cat(all_codes, dim=1).to(self.device)
        audio = decode_to_audio(merged_codes, self.codec)

        # Convert to WAV bytes
        audio_np = audio.cpu().float().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, self.codec.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ============================================================
# Fish Speech Engine — v1.5 (Legacy)
# ============================================================

class FishSpeechV15:
    """
    Fish Speech v1.5 offline engine.
    Uses DualARTransformer + Firefly GAN VQ codec.
    Requires ~4-8GB VRAM.
    """

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        import torch
        self.device = device
        self.model_dir = Path(model_dir)
        self.precision = torch.bfloat16

        logger.info(f"Loading Fish Speech v1.5 from {model_dir}...")
        t0 = time.time()

        # Load codec (VQGAN)
        from tools.vqgan.inference import load_model as load_vqgan
        codec_path = self.model_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.codec = load_vqgan("firefly_gan_vq", str(codec_path), device)

        # Load LLM
        from tools.llama.generate import (
            load_model as load_llama,
        )
        self.llama_model, self.decode_one_token = load_llama(
            checkpoint_path=str(self.model_dir),
            device=device,
            precision=self.precision,
            compile=False,
        )

        logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    def synthesize(self, text: str, ref_audio_path: str,
                   ref_text: str = "") -> bytes:
        """Synthesize speech. Returns WAV bytes."""
        import torch
        import soundfile as sf
        from tools.vqgan.inference import encode as vqgan_encode, decode as vqgan_decode
        from tools.llama.generate import generate_long

        # Encode reference audio to VQ tokens
        ref_audio, ref_sr = sf.read(ref_audio_path)
        if len(ref_audio.shape) > 1:
            ref_audio = ref_audio.mean(axis=1)
        ref_tensor = torch.from_numpy(ref_audio).float().unsqueeze(0).to(self.device)
        ref_codes = vqgan_encode(self.codec, ref_tensor, self.device)

        # Generate speech tokens
        generator = generate_long(
            model=self.llama_model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            prompt_text=[ref_text or "reference audio"],
            prompt_tokens=[ref_codes.cpu()],
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            max_new_tokens=0,
            chunk_length=300,
        )

        all_codes = []
        for response in generator:
            if response.action == "sample":
                all_codes.append(response.codes)
            elif response.action == "next":
                break

        if not all_codes:
            raise RuntimeError("No audio generated")

        # Decode to audio
        merged_codes = torch.cat(all_codes, dim=1).to(self.device)
        audio = vqgan_decode(self.codec, merged_codes)

        audio_np = audio.cpu().float().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio_np, 44100, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ============================================================
# Audio Utils
# ============================================================

def wav_duration(wav_bytes: bytes) -> float:
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        return len(audio) / sr
    except Exception:
        return 0.0


def concat_wavs(wav_list: list[bytes], pause_ms_range=(300, 800)) -> bytes:
    """Concatenate WAV files with random pauses."""
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
        pause_ms = random.randint(*pause_ms_range)
        silence = np.zeros(int(target_sr * pause_ms / 1000))
        all_audio.append(silence)

    if all_audio:
        all_audio.pop()  # Remove last silence

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
                 "lady", "woman", "여성", "женщин", "perempuan", "ผู้หญิง"]
    male_kw = ["男", "male", "他", "father", "brother", "爸", "哥", "弟",
               "man", "남성", "мужчин", "lelaki", "ผู้ชาย"]
    f = sum(1 for k in female_kw if k in desc)
    m = sum(1 for k in male_kw if k in desc)
    if f > m: return "female"
    if m > f: return "male"
    return "unknown"


# ============================================================
# Main Pipeline
# ============================================================

def process_dialogue(dlg: dict, engine, ref_manager: RefAudioManager,
                     output_dir: Path) -> Optional[dict]:
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
            logger.info(f"  Turn {turn_num} ({speaker}): {text[:60]}...")
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
        description="SwitchLingua 2.0 — Stage 2 Offline Synthesis (Direct Model Loading)")
    parser.add_argument("--input", required=True, help="Stage 1 JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--asset-dir", required=True, help="Reference audio directory")
    parser.add_argument("--model-dir", required=True, help="Fish Speech model directory")
    parser.add_argument("--version", choices=["v2", "v1.5"], default="v2",
                        help="Fish Speech version: v2 (S2-Pro, ~24GB VRAM) or v1.5 (~8GB VRAM)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Max dialogues (0=all)")
    args = parser.parse_args()

    # Init reference audio
    ref_manager = RefAudioManager(args.asset_dir)
    logger.info(f"Available languages: {ref_manager.available_languages}")

    # Init engine
    if args.version == "v2":
        engine = FishSpeechV2(args.model_dir, args.device)
    else:
        engine = FishSpeechV15(args.model_dir, args.device)

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
    elapsed = time.time() - t0
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": args.input,
            "tts_engine": f"fish-speech-{args.version}-offline",
            "model_dir": str(args.model_dir),
            "device": args.device,
            "total": len(dialogues), "success": success, "failed": failed,
            "elapsed_sec": round(elapsed, 1),
            "dialogues": manifest,
        }, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n{'='*60}\n  Done! {success}/{len(dialogues)} success, "
        f"{failed} failed, {elapsed:.1f}s ({elapsed/max(success,1):.1f}s/dlg)"
        f"\n  Output: {output_dir}\n{'='*60}"
    )


if __name__ == "__main__":
    main()
