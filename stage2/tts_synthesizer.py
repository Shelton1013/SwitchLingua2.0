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

    # Max characters per TTS request. CosyVoice 3 struggles with long text
    # (only generates audio for the tail end). Split into sentences and
    # synthesize individually, then concatenate.
    _MAX_CHARS_PER_REQUEST = 35

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences by punctuation, keeping each chunk
        under _MAX_CHARS_PER_REQUEST characters."""
        import re
        # Split on sentence-ending punctuation (keep the punctuation attached)
        parts = re.split(r'(?<=[。！？.!?，,、；;])', text)
        parts = [p.strip() for p in parts if p.strip()]

        # Merge very short fragments together
        merged = []
        buf = ""
        for p in parts:
            if len(buf) + len(p) <= CosyVoiceSynthesizer._MAX_CHARS_PER_REQUEST:
                buf += p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        return merged if merged else [text]

    def synthesize(self, text: str, reference_audio_path: str,
                   reference_text: str = "",
                   output_path: Optional[str] = None,
                   lang_code: str = "") -> bytes:
        """Synthesize speech for given text using reference audio for voice cloning.

        Long texts are automatically split into sentences and synthesized
        individually to avoid CosyVoice 3's truncation issue.

        Args:
            text: Text to synthesize (can be multilingual/code-switching)
            reference_audio_path: Path to reference WAV file (3-10s)
            reference_text: Transcript of reference audio (optional but recommended)
            output_path: If provided, save WAV to this path
            lang_code: L1 language code (unused, kept for API compatibility)

        Returns:
            Raw WAV bytes

        Raises:
            RuntimeError: If synthesis fails after all retries
        """
        # Split long text into manageable sentences
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            logger.info(
                "Text too long (%d chars), split into %d sentences",
                len(text), len(sentences),
            )

        # Synthesize each sentence and collect PCM data
        all_pcm = []
        for i, sentence in enumerate(sentences):
            pcm = self._synthesize_one(
                sentence, reference_audio_path, reference_text,
            )
            all_pcm.append(pcm)
            if len(sentences) > 1:
                logger.info(
                    "  Sentence %d/%d (%d chars): %d bytes PCM",
                    i + 1, len(sentences), len(sentence), len(pcm),
                )

        # Combine all PCM data into one WAV
        combined_pcm = b"".join(all_pcm)
        wav_bytes = self._pcm_to_wav(combined_pcm)

        logger.info(
            "TTS synthesis OK — %d bytes, %.2fs duration",
            len(wav_bytes), self.get_wav_duration(wav_bytes),
        )

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(wav_bytes)
            logger.info("Saved synthesized audio to %s", out)

        return wav_bytes

    def _synthesize_one(self, text: str, reference_audio_path: str,
                        reference_text: str = "") -> bytes:
        """Synthesize a single short text segment. Returns raw PCM bytes."""
        ref_path = Path(reference_audio_path)

        PROMPT_MARKER = "<|endofprompt|>"
        if reference_text.strip():
            prompt_text = reference_text
        else:
            prompt_text = "这是一段参考语音。"
        if PROMPT_MARKER not in prompt_text:
            prompt_text = f"You are a helpful assistant.{PROMPT_MARKER}{prompt_text}"

        url = f"{self.base_url}/inference_zero_shot"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "TTS request attempt %d/%d — text='%s' (%d chars)",
                    attempt, self.max_retries, text[:30], len(text),
                )

                with open(ref_path, "rb") as audio_fh:
                    files = {
                        "prompt_wav": (ref_path.name, audio_fh, "audio/wav"),
                    }
                    data = {
                        "tts_text": text,
                        "prompt_text": prompt_text,
                    }

                    resp = self._session.post(
                        url,
                        data=data,
                        files=files,
                        timeout=self.timeout,
                        stream=True,
                    )

                resp.raise_for_status()

                chunks = []
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                raw_bytes = b"".join(chunks)

                if not raw_bytes:
                    raise RuntimeError("Server returned empty audio response")

                # Strip WAV header if present — we need raw PCM for concatenation
                if raw_bytes[:4] == b'RIFF':
                    raw_bytes = self._wav_to_pcm(raw_bytes)

                return raw_bytes

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
    def _wav_to_pcm(wav_bytes: bytes) -> bytes:
        """Extract raw PCM data from a WAV container."""
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                return wf.readframes(wf.getnframes())

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
