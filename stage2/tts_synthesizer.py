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

    def synthesize(self, text: str, reference_audio_path: str,
                   reference_text: str = "",
                   output_path: Optional[str] = None) -> bytes:
        """Synthesize speech for given text using reference audio for voice cloning.

        Args:
            text: Text to synthesize (can be multilingual/code-switching)
            reference_audio_path: Path to reference WAV file (3-10s)
            reference_text: Transcript of reference audio (optional but recommended)
            output_path: If provided, save WAV to this path

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

        # Use a generic placeholder if reference_text is empty
        prompt_text = reference_text if reference_text.strip() else "这是一段参考语音。"

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

                    resp = self._session.post(
                        url,
                        data=data,
                        files=files,
                        timeout=self.timeout,
                        stream=True,
                    )

                resp.raise_for_status()

                # Collect the streamed WAV chunks
                chunks = []
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                wav_bytes = b"".join(chunks)

                if not wav_bytes:
                    raise RuntimeError("Server returned empty audio response")

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

    def synthesize_turn(self, text: str, voice_profile,
                        output_dir: str, turn_num: int,
                        speaker_name: str) -> dict:
        """Synthesize one dialogue turn and save to file.

        Args:
            text: Turn text
            voice_profile: VoiceProfile object (has .audio_file attribute)
            output_dir: Directory to save the wav file
            turn_num: Turn number (for filename)
            speaker_name: "A" or "B" (for filename)

        Returns:
            dict with keys: audio_file, duration_sec
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"turn_{turn_num}_{speaker_name}.wav"
        output_path = str(out_dir / filename)

        # Use the transcript stored on the voice profile if available,
        # otherwise fall back to empty (synthesize will use placeholder).
        ref_text = getattr(voice_profile, "transcript", "")

        wav_bytes = self.synthesize(
            text=text,
            reference_audio_path=voice_profile.audio_file,
            reference_text=ref_text,
            output_path=output_path,
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
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def get_wav_duration(wav_bytes: bytes) -> float:
        """Get duration in seconds from WAV bytes."""
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
