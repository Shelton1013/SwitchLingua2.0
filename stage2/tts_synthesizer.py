"""
SwitchLingua 2.0 — Stage 2: Fish Speech S2 TTS Synthesizer

Calls Fish Speech S2 API server for zero-shot voice cloning TTS.
Supports all 80+ languages including zh, yue, en, ja, fr, es, hi, ms.

Requires:
    pip install ormsgpack requests

Usage:
    synth = FishSpeechSynthesizer(base_url="http://localhost:8080")
    wav = synth.synthesize("你好世界", "ref.wav", "参考音频的文字", output_path="out.wav")
"""

import io
import re
import wave
import time
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("tts_synthesizer")


class FishSpeechSynthesizer:
    """Client for Fish Speech S2 API server."""

    def __init__(self, base_url: str = "http://localhost:8080",
                 timeout: int = 120, max_retries: int = 3,
                 temperature: float = 0.8, top_p: float = 0.8,
                 repetition_penalty: float = 1.1, **kwargs):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self._session = requests.Session()

    def synthesize(self, text: str, reference_audio_path: str,
                   reference_text: str = "",
                   output_path: Optional[str] = None,
                   lang_code: str = "") -> bytes:
        """Synthesize speech using Fish Speech S2 zero-shot voice cloning.

        Args:
            text: Text to synthesize (any language, including code-switching)
            reference_audio_path: Path to reference audio file (3-10s)
            reference_text: Transcript of reference audio (important for quality)
            output_path: If provided, save WAV to this path
            lang_code: Unused (Fish Speech auto-detects language)

        Returns:
            WAV bytes
        """
        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        ref_audio_bytes = ref_path.read_bytes()

        try:
            import ormsgpack
            use_msgpack = True
        except ImportError:
            use_msgpack = False

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "TTS attempt %d/%d — text='%s...' (%d chars), ref=%s",
                    attempt, self.max_retries, text[:30], len(text),
                    ref_path.name,
                )

                if use_msgpack:
                    request_data = {
                        "text": text,
                        "references": [
                            {
                                "audio": ref_audio_bytes,
                                "text": reference_text or "reference audio",
                            }
                        ],
                        "format": "wav",
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "repetition_penalty": self.repetition_penalty,
                        "normalize": True,
                    }
                    resp = self._session.post(
                        f"{self.base_url}/v1/tts",
                        data=ormsgpack.packb(request_data),
                        headers={"Content-Type": "application/msgpack"},
                        timeout=self.timeout,
                    )
                else:
                    resp = self._session.post(
                        f"{self.base_url}/v1/tts",
                        json={
                            "text": text,
                            "format": "wav",
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "repetition_penalty": self.repetition_penalty,
                        },
                        timeout=self.timeout,
                    )

                resp.raise_for_status()
                wav_bytes = resp.content

                if not wav_bytes or len(wav_bytes) < 100:
                    raise RuntimeError(f"Too little data ({len(wav_bytes)} bytes)")

                duration = self.get_wav_duration(wav_bytes)
                logger.info("TTS synthesis OK — %.2fs duration", duration)

                if output_path is not None:
                    out = Path(output_path)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(wav_bytes)

                return wav_bytes

            except Exception as exc:
                last_error = exc
                logger.warning("TTS attempt %d/%d failed: %s", attempt, self.max_retries, exc)
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
        """Synthesize one dialogue turn and save to file."""
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
        return {"audio_file": filename, "duration_sec": round(duration, 3)}

    def check_health(self) -> bool:
        """Check if Fish Speech server is responding."""
        try:
            resp = self._session.get(f"{self.base_url}/v1/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def get_wav_duration(wav_bytes: bytes) -> float:
        """Get duration in seconds from WAV bytes."""
        try:
            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    return wf.getnframes() / wf.getframerate()
        except Exception:
            return len(wav_bytes) / (44100 * 2)


# Backwards-compatible alias
CosyVoiceSynthesizer = FishSpeechSynthesizer
