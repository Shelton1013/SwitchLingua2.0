"""
SwitchLingua 2.0 — Stage 2: CosyVoice 3 TTS Synthesizer

Wraps the CosyVoice 3 FastAPI server for zero-shot voice cloning TTS.
"""

import io
import time
import wave
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tts_synthesizer")


class CosyVoiceSynthesizer:
    """Client for CosyVoice 3 FastAPI TTS server."""

    def __init__(self, base_url: str = "http://localhost:50000",
                 timeout: int = 60, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    # Language instruction templates for CosyVoice instruct2 endpoint.
    # Tells the model what language/dialect to use for pronunciation.
    _LANG_INSTRUCTIONS = {
        "zh": "请用普通话朗读以下文本，英文部分用英语发音。",
        "yue": "请用粤语朗读以下文本，英文部分用英语发音。",
        "ja": "以下のテキストを日本語で読んでください。英語の部分は英語で発音してください。",
        "fr": "Lisez le texte suivant en français. Prononcez les mots anglais en anglais.",
        "es": "Lea el siguiente texto en español. Pronuncie las palabras en inglés como inglés.",
        "hi": "कृपया इस टेक्स्ट को हिंदी में पढ़ें। अंग्रेज़ी शब्दों को अंग्रेज़ी में बोलें।",
        "ms": "Sila baca teks ini dalam Bahasa Melayu. Sebut perkataan Inggeris dalam Bahasa Inggeris.",
        "min": "请用闽南语朗读以下文本，英文部分用英语发音。",
        "en": "Please read the following text in English.",
    }

    def synthesize(self, text: str, reference_audio_path: str,
                   reference_text: str = "",
                   output_path: Optional[str] = None,
                   lang_code: str = "") -> bytes:
        """Synthesize speech for given text using reference audio for voice cloning.

        Args:
            text: Text to synthesize (can be multilingual/code-switching)
            reference_audio_path: Path to reference WAV file (3-10s)
            reference_text: Transcript of reference audio (optional but recommended)
            output_path: If provided, save WAV to this path
            lang_code: L1 language code (e.g. "zh", "yue", "fr"). When provided,
                uses the instruct2 endpoint with a language-specific instruction
                to ensure correct pronunciation. When empty, uses zero_shot.

        Returns:
            Raw WAV bytes

        Raises:
            RuntimeError: If synthesis fails after all retries
        """
        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio_path}"
            )

        # CosyVoice 3 requires <|endofprompt|> marker in prompt_text.
        # Format: "instruction<|endofprompt|>transcript of reference audio"
        PROMPT_MARKER = "<|endofprompt|>"
        if reference_text.strip():
            prompt_text = reference_text
        else:
            prompt_text = "这是一段参考语音。"
        # Ensure the marker is present (CosyVoice 3 will error without it)
        if PROMPT_MARKER not in prompt_text:
            prompt_text = f"You are a helpful assistant.{PROMPT_MARKER}{prompt_text}"

        # Use instruct2 for language-specific pronunciation (e.g. Cantonese vs Mandarin),
        # fall back to zero_shot for auto-detect.
        instruct_text = self._LANG_INSTRUCTIONS.get(lang_code, "")
        if instruct_text:
            url = f"{self.base_url}/inference_instruct2"
        else:
            url = f"{self.base_url}/inference_zero_shot"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "TTS request attempt %d/%d — text length=%d, ref=%s",
                    attempt, self.max_retries, len(text), ref_path.name,
                )

                with open(ref_path, "rb") as audio_fh:
                    files = {
                        "prompt_wav": (ref_path.name, audio_fh, "audio/wav"),
                    }
                    data = {
                        "tts_text": text,
                        "prompt_text": prompt_text,
                    }
                    if instruct_text:
                        data["instruct_text"] = instruct_text

                    resp = self._session.post(
                        url,
                        data=data,
                        files=files,
                        timeout=self.timeout,
                        stream=True,
                    )

                resp.raise_for_status()

                # Collect the streamed audio chunks
                chunks = []
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                raw_bytes = b"".join(chunks)

                if not raw_bytes:
                    raise RuntimeError("Server returned empty audio response")

                # Official CosyVoice server streams raw int16 PCM at 24kHz.
                # If response is already WAV (has RIFF header), use as-is.
                # Otherwise wrap raw PCM into a WAV container.
                if raw_bytes[:4] == b'RIFF':
                    wav_bytes = raw_bytes
                else:
                    wav_bytes = self._pcm_to_wav(raw_bytes)

                logger.info(
                    "TTS synthesis OK — %d bytes, %.2fs duration",
                    len(wav_bytes),
                    self.get_wav_duration(wav_bytes),
                )

                # Save to disk if requested
                if output_path is not None:
                    out = Path(output_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(wav_bytes)
                    logger.info("Saved synthesized audio to %s", out)

                return wav_bytes

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "TTS attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    logger.info("Retrying in %ds...", backoff)
                    time.sleep(backoff)

        raise RuntimeError(
            f"TTS synthesis failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def synthesize_turn(self, text: str, reference_audio_path: str,
                        output_dir: str, turn_num: int,
                        speaker_name: str,
                        lang_code: str = "") -> dict:
        """Synthesize one dialogue turn and save to file.

        Args:
            text: Turn text
            reference_audio_path: Path to the reference WAV for voice cloning
            output_dir: Directory to save the wav file
            turn_num: Turn number (for filename)
            speaker_name: "A" or "B" (for filename)
            lang_code: L1 language code for pronunciation (e.g. "zh", "yue")

        Returns:
            dict with keys: audio_file, duration_sec
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"turn_{turn_num}_{speaker_name}.wav"
        output_path = str(out_dir / filename)

        wav_bytes = self.synthesize(
            text=text,
            reference_audio_path=reference_audio_path,
            output_path=output_path,
            lang_code=lang_code,
        )

        duration = self.get_wav_duration(wav_bytes)

        return {
            "audio_file": output_path,
            "duration_sec": round(duration, 3),
        }

    def check_health(self) -> bool:
        """Check if CosyVoice server is responding."""
        try:
            resp = self._session.get(f"{self.base_url}/", timeout=5)
            # 200 = our server, 404 = official server (no / endpoint but alive)
            return resp.status_code in (200, 404)
        except Exception:
            return False

    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000,
                    channels: int = 1, sample_width: int = 2) -> bytes:
        """Wrap raw PCM int16 bytes into a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    @staticmethod
    def get_wav_duration(wav_bytes: bytes) -> float:
        """Get duration in seconds from WAV bytes."""
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
