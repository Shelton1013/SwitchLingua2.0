"""
SwitchLingua 2.0 — Stage 2: CosyVoice 3 TTS Synthesizer

Direct SDK integration with CosyVoice for zero-shot voice cloning TTS.
No HTTP server needed — loads model directly in the pipeline process.

Usage:
    synth = CosyVoiceSynthesizer(model_dir="/data/models/Fun-CosyVoice3-0.5B-2512")
    wav = synth.synthesize("你好世界", "ref_audio.wav", output_path="out.wav")
"""

import io
import re
import wave
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tts_synthesizer")


class CosyVoiceSynthesizer:
    """Direct CosyVoice SDK synthesizer (no HTTP server needed)."""

    PROMPT_MARKER = "<|endofprompt|>"

    def __init__(self, model_dir: str = "", base_url: str = "",
                 timeout: int = 60, max_retries: int = 3):
        """Initialize synthesizer.

        Args:
            model_dir: Path to CosyVoice model. If provided, loads model
                directly via SDK (recommended).
            base_url: DEPRECATED. Kept for backwards compatibility but ignored.
            timeout: Not used in SDK mode.
            max_retries: Number of retries on synthesis failure.
        """
        self.max_retries = max_retries
        self._model = None
        self._sample_rate = 24000

        if model_dir:
            self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        """Load CosyVoice model via SDK."""
        from cosyvoice.cli.cosyvoice import AutoModel
        logger.info(f"Loading CosyVoice model from {model_dir}...")
        self._model = AutoModel(model_dir=model_dir)
        self._sample_rate = self._model.sample_rate
        logger.info(f"Model loaded: {type(self._model).__name__}, sr={self._sample_rate}")

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
        """Synthesize speech using CosyVoice SDK directly.

        Args:
            text: Text to synthesize (can be multilingual/code-switching)
            reference_audio_path: Path to reference WAV file (3-10s)
            reference_text: Transcript of reference audio (optional)
            output_path: If provided, save WAV to this path
            lang_code: Unused, kept for API compatibility

        Returns:
            WAV bytes

        Raises:
            RuntimeError: If synthesis fails after all retries
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Pass model_dir to constructor.")

        ref_path = str(Path(reference_audio_path))

        # Build prompt_text with <|endofprompt|> marker
        if reference_text.strip():
            prompt_text = reference_text
        else:
            prompt_text = "这是一段参考语音。"
        if self.PROMPT_MARKER not in prompt_text:
            prompt_text = f"You are a helpful assistant.{self.PROMPT_MARKER}{prompt_text}"

        # Split long text into sentences to avoid CosyVoice truncation.
        # CosyVoice 3 struggles with text > ~30 chars, producing incomplete audio.
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            logger.info(
                "Splitting text (%d chars) into %d segments for synthesis",
                len(text), len(sentences),
            )

        import torch
        all_pcm = []

        for si, sentence in enumerate(sentences):
            pcm = self._synthesize_one(sentence, ref_path, prompt_text)
            all_pcm.append(pcm)
            if len(sentences) > 1:
                logger.info(
                    "  Segment %d/%d: '%s' (%d chars)",
                    si + 1, len(sentences), sentence[:20], len(sentence),
                )

        # Combine all segments into one WAV
        combined_pcm = b"".join(all_pcm)
        wav_bytes = self._pcm_to_wav(combined_pcm, sample_rate=self._sample_rate)

        duration = self.get_wav_duration(wav_bytes)
        logger.info("TTS synthesis OK — %.2fs total duration", duration)

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(wav_bytes)

        return wav_bytes

    # Max chars per synthesis call. CosyVoice 3 truncates beyond this.
    _MAX_SEGMENT_CHARS = 30

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text at punctuation, merging short fragments."""
        parts = re.split(r'(?<=[。！？.!?，,、；;：:])', text)
        parts = [p.strip() for p in parts if p.strip()]

        merged = []
        buf = ""
        for p in parts:
            if len(buf) + len(p) <= CosyVoiceSynthesizer._MAX_SEGMENT_CHARS:
                buf += p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        return merged if merged else [text]

    def _synthesize_one(self, text: str, ref_path: str,
                        prompt_text: str) -> bytes:
        """Synthesize one short segment. Returns raw PCM int16 bytes."""
        import torch
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                all_audio = []
                for result in self._model.inference_zero_shot(
                    text, prompt_text, ref_path, stream=False,
                ):
                    all_audio.append(result["tts_speech"])

                if not all_audio:
                    raise RuntimeError("Model returned no audio")

                full_audio = torch.cat(all_audio, dim=-1)
                audio_np = full_audio.cpu().numpy().flatten()
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                return audio_int16.tobytes()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "TTS segment attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                import time
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
                        reference_text: str = "",
                        lang_code: str = "") -> dict:
        """Synthesize one dialogue turn and save to file.

        Args:
            text: Turn text
            reference_audio_path: Path to the reference WAV for voice cloning
            output_dir: Directory to save the wav file
            turn_num: Turn number (for filename)
            speaker_name: "A" or "B" (for filename)
            reference_text: Transcript of reference audio (critical for quality)
            lang_code: L1 language code (unused, kept for compatibility)

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
            reference_text=reference_text,
            output_path=output_path,
            lang_code=lang_code,
        )

        duration = self.get_wav_duration(wav_bytes)

        return {
            "audio_file": filename,  # relative name, not full path
            "duration_sec": round(duration, 3),
        }

    def check_health(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

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
