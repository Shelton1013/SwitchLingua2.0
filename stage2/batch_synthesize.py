"""
SwitchLingua 2.0 — Stage 2: Batch Speech Synthesis

Reads Stage 1 JSONL, synthesizes each turn via Fish Speech S2.
Automatically selects reference audio by language from an asset folder.

Asset folder structure:
    asset/
    ├── zh_male_01.wav          # Files starting with language code
    ├── zh_female_01.wav        # Multiple files per language = random selection
    ├── yue_male_01.wav
    ├── en_male_01.wav
    ├── ja_female_01.wav
    └── ...

Each file must have a matching .txt with the transcript:
    asset/
    ├── zh_male_01.wav
    ├── zh_male_01.txt          # Contains: "这段音频的文字转录"
    ├── yue_male_01.wav
    ├── yue_male_01.txt         # Contains: "呢段音頻嘅文字"
    └── ...

Usage:
    python stage2/batch_synthesize.py \
        --input output/yue-en.jsonl \
        --output output/stage2/yue_en/ \
        --asset-dir stage2/asset \
        --fish-url http://localhost:8080 \
        --limit 5
"""

import io
import json
import time
import wave
import struct
import random
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import requests

logger = logging.getLogger("batch_synthesize")


# ============================================================
# Reference Audio Manager
# ============================================================

class RefAudioManager:
    """Manages reference audio files organized by language code.

    Scans an asset directory for audio files named {lang_code}_{anything}.wav
    and their matching .txt transcript files. At synthesis time, randomly
    picks one reference audio for the requested language.
    """

    def __init__(self, asset_dir: str):
        self.asset_dir = Path(asset_dir)
        if not self.asset_dir.exists():
            raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

        # Index: lang_code -> [(audio_path, transcript), ...]
        self._index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._scan()

    def _scan(self):
        """Scan asset directory and build language index.

        Supports two naming conventions:
          - {lang}_{gender}_{id}.wav  (e.g. fr_male_01.wav)
          - {lang}_{id}.wav           (e.g. fr_01.wav, gender unknown)

        Each .wav must have a matching .txt with the transcript.
        """
        audio_exts = {".wav", ".mp3", ".flac", ".ogg"}

        for f in sorted(self.asset_dir.iterdir()):
            if f.suffix.lower() not in audio_exts:
                continue

            parts = f.stem.split("_")
            if not parts:
                continue
            lang_code = parts[0].lower()

            # Infer gender from filename: {lang}_{gender}_{...}
            gender = "unknown"
            if len(parts) >= 2 and parts[1].lower() in ("male", "female"):
                gender = parts[1].lower()

            # Look for matching transcript file
            txt_path = f.with_suffix(".txt")
            if txt_path.exists():
                transcript = txt_path.read_text(encoding="utf-8").strip()
            else:
                transcript = ""
                logger.warning(f"No transcript for {f.name} (expected {txt_path.name})")

            self._index[lang_code].append((str(f), transcript, gender))

        # Log summary
        for lang, refs in sorted(self._index.items()):
            genders = [r[2] for r in refs]
            logger.info(f"  {lang}: {len(refs)} ref(s) — {', '.join(genders)}")

        if not self._index:
            logger.warning(f"No reference audio found in {self.asset_dir}")

    def get_ref(self, lang_code: str, gender: str = "unknown") -> tuple[str, str]:
        """Get a (audio_path, transcript) for the given language and gender.

        Match priority: exact lang+gender > same lang any gender > 'en' > any.
        """
        lang_code = lang_code.lower()
        gender = gender.lower()

        refs = self._index.get(lang_code, [])
        if refs:
            # Try gender match first
            if gender != "unknown":
                gender_matched = [r for r in refs if r[2] == gender]
                if gender_matched:
                    chosen = random.choice(gender_matched)
                    return chosen[0], chosen[1]
            # Fallback: any gender for this language
            chosen = random.choice(refs)
            return chosen[0], chosen[1]

        # Fallback: try English
        if "en" in self._index:
            logger.warning(f"No ref audio for '{lang_code}', falling back to 'en'")
            chosen = random.choice(self._index["en"])
            return chosen[0], chosen[1]

        # Fallback: any available
        all_refs = [ref for refs in self._index.values() for ref in refs]
        if all_refs:
            logger.warning(f"No ref audio for '{lang_code}', using random fallback")
            chosen = random.choice(all_refs)
            return chosen[0], chosen[1]

        raise RuntimeError(f"No reference audio available for '{lang_code}' or any fallback")

    @property
    def available_languages(self) -> list[str]:
        return sorted(self._index.keys())


# ============================================================
# TTS Client
# ============================================================

class FishTTS:
    """Minimal Fish Speech S2 client."""

    def __init__(self, base_url: str, timeout: int = 600, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._ref_cache = {}

    def _get_ref_bytes(self, ref_path: str) -> bytes:
        if ref_path not in self._ref_cache:
            self._ref_cache[ref_path] = Path(ref_path).read_bytes()
        return self._ref_cache[ref_path]

    def synthesize(self, text: str, ref_audio: str, ref_text: str) -> bytes:
        try:
            import ormsgpack
        except ImportError:
            raise ImportError("pip install ormsgpack")

        ref_bytes = self._get_ref_bytes(ref_audio)
        request_data = {
            "text": text,
            "references": [{"audio": ref_bytes, "text": ref_text or "reference audio"}],
            "format": "wav",
            "temperature": 0.8,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "normalize": True,
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    f"{self.base_url}/v1/tts",
                    data=ormsgpack.packb(request_data),
                    headers={"Content-Type": "application/msgpack"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                if len(resp.content) < 100:
                    raise RuntimeError(f"Response too small: {len(resp.content)} bytes")
                return resp.content
            except Exception as e:
                last_err = e
                logger.warning("TTS attempt %d/%d failed: %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(f"TTS failed after {self.max_retries} retries: {last_err}")

    def health(self) -> bool:
        try:
            return self._session.get(f"{self.base_url}/v1/health", timeout=5).status_code == 200
        except Exception:
            return False


# ============================================================
# Audio utils
# ============================================================

def wav_duration(wav_bytes: bytes) -> float:
    try:
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def wav_to_pcm(wav_bytes: bytes) -> tuple:
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def concat_wavs(wav_list: list, pause_ms_range=(300, 800)) -> bytes:
    if not wav_list:
        return b""

    first_pcm, sr, ch, sw = wav_to_pcm(wav_list[0])
    all_pcm = [first_pcm]

    for wav_bytes in wav_list[1:]:
        pause_ms = random.randint(*pause_ms_range)
        n_silence = int(sr * pause_ms / 1000)
        silence = struct.pack(f"<{n_silence * ch}h", *([0] * n_silence * ch))
        all_pcm.append(silence)
        pcm, _, _, _ = wav_to_pcm(wav_bytes)
        all_pcm.append(pcm)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(b"".join(all_pcm))
    return buf.getvalue()


# ============================================================
# Pipeline
# ============================================================

def _infer_gender(persona_desc: str) -> str:
    """Infer gender from persona description."""
    desc = persona_desc.lower()
    female_kw = ["女", "female", "她", "mother", "sister", "妈", "姐", "妹",
                 "lady", "woman", "여성", "أنثى", "महिला", "perempuan", "babae"]
    male_kw = ["男", "male", "他", "father", "brother", "爸", "哥", "弟",
               "man", "남성", "ذكر", "पुरुष", "lelaki", "lalaki"]
    f = sum(1 for k in female_kw if k in desc)
    m = sum(1 for k in male_kw if k in desc)
    if f > m:
        return "female"
    if m > f:
        return "male"
    return "unknown"


def process_dialogue(dlg: dict, tts: FishTTS, ref_manager: RefAudioManager,
                     output_dir: Path) -> Optional[dict]:
    dlg_id = dlg["dialogue_id"]
    dlg_dir = output_dir / dlg_id
    dlg_dir.mkdir(parents=True, exist_ok=True)

    lang_pair = dlg.get("language_pair", [])
    l1 = lang_pair[0] if lang_pair else "en"

    # Infer gender for each speaker from persona description
    gender_a = _infer_gender(dlg.get("speaker_a", {}).get("persona_description", ""))
    gender_b = _infer_gender(dlg.get("speaker_b", {}).get("persona_description", ""))
    # If both same gender, try to differentiate
    if gender_a == gender_b and gender_a == "unknown":
        gender_a, gender_b = "male", "female"

    # Pick different reference audio for A and B
    ref_a, ref_text_a = ref_manager.get_ref(l1, gender_a)
    ref_b, ref_text_b = ref_manager.get_ref(l1, gender_b)
    # If they got the same file, try to get a different one for B
    if ref_a == ref_b:
        opposite = "female" if gender_a == "male" else "male"
        ref_b, ref_text_b = ref_manager.get_ref(l1, opposite)

    voice_map = {
        "A": (ref_a, ref_text_a),
        "B": (ref_b, ref_text_b),
    }
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
            wav_bytes = tts.synthesize(text, ref_audio, ref_text)
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

    # Assemble
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
        "full_audio": full_audio_file, "full_duration_sec": round(full_duration, 2),
    }
    with open(dlg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 — Batch Speech Synthesis (Fish Speech S2)")
    parser.add_argument("--input", required=True, help="Stage 1 JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--asset-dir", required=True,
                        help="Directory with reference audio files named {lang}_{label}.wav + .txt")
    parser.add_argument("--fish-url", default="http://localhost:8080")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Init
    ref_manager = RefAudioManager(args.asset_dir)
    logger.info(f"Available languages: {ref_manager.available_languages}")

    tts = FishTTS(args.fish_url, timeout=args.timeout)
    if not tts.health():
        logger.error(f"Fish Speech not responding at {args.fish_url}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
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

        result = process_dialogue(dlg, tts, ref_manager, output_dir)
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

    # Manifest
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": args.input, "tts_engine": "fish-speech-s2",
            "asset_dir": str(args.asset_dir),
            "available_languages": ref_manager.available_languages,
            "total": len(dialogues), "success": success, "failed": failed,
            "elapsed_sec": round(time.time() - t0, 1),
            "dialogues": manifest,
        }, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n{'='*60}\n  Done! {success}/{len(dialogues)} success, "
        f"{failed} failed, {time.time()-t0:.1f}s\n  Output: {output_dir}\n{'='*60}"
    )


if __name__ == "__main__":
    main()
