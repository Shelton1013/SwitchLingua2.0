"""
SwitchLingua 2.0 — Stage 2: Audio Assembler

Concatenates per-turn WAV files into a full dialogue audio,
inserting natural pauses between turns.
"""

import io
import wave
import struct
import random
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("audio_assembler")

# Target format: 16kHz, mono, 16-bit PCM
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit


class AudioAssembler:
    """Assembles per-turn WAV files into complete dialogue audio."""

    def __init__(self, pause_range: tuple = (300, 800),
                 target_sample_rate: int = TARGET_SAMPLE_RATE):
        """
        Args:
            pause_range: (min_ms, max_ms) range for random pause between turns
            target_sample_rate: Output sample rate in Hz
        """
        self.pause_min_ms, self.pause_max_ms = pause_range
        self.sample_rate = target_sample_rate

    def _generate_silence(self, duration_ms: int) -> bytes:
        """Generate silence (zero samples) for given duration."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return struct.pack(f'<{num_samples}h', *([0] * num_samples))

    def _read_wav_pcm(self, wav_path: str) -> tuple:
        """Read WAV file and return (pcm_data, sample_rate, channels, sample_width)."""
        with wave.open(str(wav_path), 'rb') as wf:
            return (wf.readframes(wf.getnframes()),
                    wf.getframerate(), wf.getnchannels(), wf.getsampwidth())

    def _resample_if_needed(self, pcm_data: bytes, src_rate: int,
                            src_channels: int, src_width: int) -> bytes:
        """Convert to target format if different.

        Handles:
          - Stereo to mono conversion (average of L and R channels)
          - Sample width conversion (8/24/32-bit to 16-bit)
          - Sample rate conversion via linear interpolation

        Args:
            pcm_data: Raw PCM bytes from wave.readframes
            src_rate: Source sample rate
            src_channels: Source channel count (1=mono, 2=stereo)
            src_width: Source sample width in bytes (1=8-bit, 2=16-bit, 3=24-bit, 4=32-bit)

        Returns:
            PCM bytes in target format (16kHz, mono, 16-bit)
        """
        # --- Step 1: Decode raw bytes into a list of integer samples ---
        num_bytes = len(pcm_data)
        total_samples = num_bytes // src_width  # total across all channels

        if src_width == 1:
            # 8-bit unsigned
            samples = list(struct.unpack(f'<{total_samples}B', pcm_data))
            # Convert unsigned 8-bit [0..255] to signed [-128..127]
            samples = [s - 128 for s in samples]
        elif src_width == 2:
            # 16-bit signed little-endian
            samples = list(struct.unpack(f'<{total_samples}h', pcm_data))
        elif src_width == 3:
            # 24-bit signed little-endian — no struct format, decode manually
            samples = []
            for i in range(0, num_bytes, 3):
                # Read 3 bytes, sign-extend to 32-bit
                lo = pcm_data[i]
                mid = pcm_data[i + 1]
                hi = pcm_data[i + 2]
                value = lo | (mid << 8) | (hi << 16)
                if value >= 0x800000:
                    value -= 0x1000000
                samples.append(value)
        elif src_width == 4:
            # 32-bit signed little-endian
            samples = list(struct.unpack(f'<{total_samples}i', pcm_data))
        else:
            raise ValueError(f"Unsupported sample width: {src_width} bytes")

        # --- Step 2: Stereo to mono (average L and R) ---
        if src_channels == 2:
            mono_samples = []
            for i in range(0, len(samples), 2):
                avg = (samples[i] + samples[i + 1]) // 2
                mono_samples.append(avg)
            samples = mono_samples
        elif src_channels != 1:
            # For >2 channels, take the first channel
            mono_samples = []
            for i in range(0, len(samples), src_channels):
                mono_samples.append(samples[i])
            samples = mono_samples

        # --- Step 3: Normalize to 16-bit range ---
        if src_width == 1:
            # 8-bit signed [-128..127] -> 16-bit [-32768..32767]
            samples = [s * 256 for s in samples]
        elif src_width == 2:
            pass  # already 16-bit
        elif src_width == 3:
            # 24-bit [-8388608..8388607] -> 16-bit: shift right by 8
            samples = [s >> 8 for s in samples]
        elif src_width == 4:
            # 32-bit [-2147483648..2147483647] -> 16-bit: shift right by 16
            samples = [s >> 16 for s in samples]

        # Clamp to 16-bit range
        samples = [max(-32768, min(32767, s)) for s in samples]

        # --- Step 4: Resample if sample rate differs ---
        if src_rate != self.sample_rate:
            src_len = len(samples)
            ratio = src_rate / self.sample_rate
            dst_len = int(src_len / ratio)
            resampled = []
            for i in range(dst_len):
                src_pos = i * ratio
                idx = int(src_pos)
                frac = src_pos - idx
                if idx + 1 < src_len:
                    # Linear interpolation between two adjacent samples
                    value = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
                else:
                    value = samples[idx] if idx < src_len else 0
                resampled.append(int(round(value)))
            samples = resampled
            # Clamp again after interpolation
            samples = [max(-32768, min(32767, s)) for s in samples]

        # --- Step 5: Pack back to 16-bit PCM bytes ---
        return struct.pack(f'<{len(samples)}h', *samples)

    def assemble(self, turn_wav_paths: list, output_path: str) -> float:
        """Assemble turn WAVs into one dialogue audio file.

        Args:
            turn_wav_paths: Ordered list of WAV file paths for each turn
            output_path: Path to write the assembled WAV

        Returns:
            Total duration in seconds
        """
        if not turn_wav_paths:
            raise ValueError("No turn WAV paths provided")

        all_pcm = bytearray()
        num_turns = len(turn_wav_paths)

        for idx, wav_path in enumerate(turn_wav_paths):
            logger.debug("Reading turn %d: %s", idx, wav_path)
            try:
                pcm_data, src_rate, src_channels, src_width = self._read_wav_pcm(wav_path)
            except Exception as e:
                logger.error("Failed to read %s: %s", wav_path, e)
                raise

            # Resample / convert to target format
            if (src_rate != self.sample_rate or
                    src_channels != TARGET_CHANNELS or
                    src_width != TARGET_SAMPLE_WIDTH):
                pcm_data = self._resample_if_needed(
                    pcm_data, src_rate, src_channels, src_width)

            all_pcm.extend(pcm_data)

            # Insert random pause between turns (not after the last one)
            if idx < num_turns - 1:
                pause_ms = random.randint(self.pause_min_ms, self.pause_max_ms)
                logger.debug("Inserting %d ms pause after turn %d", pause_ms, idx)
                all_pcm.extend(self._generate_silence(pause_ms))

        # Write the assembled WAV
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output), 'wb') as wf:
            wf.setnchannels(TARGET_CHANNELS)
            wf.setsampwidth(TARGET_SAMPLE_WIDTH)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(all_pcm))

        total_samples = len(all_pcm) // TARGET_SAMPLE_WIDTH
        total_duration = total_samples / self.sample_rate
        logger.info("Assembled %d turns -> %s (%.2f s)",
                     num_turns, output_path, total_duration)
        return total_duration

    def assemble_dialogue(self, dialogue_dir: str, turn_files: list) -> dict:
        """Assemble a dialogue's turns into dialogue_full.wav.

        Args:
            dialogue_dir: Directory containing turn WAV files
            turn_files: List of turn filenames in order

        Returns:
            dict with: full_audio filename, full_duration_sec
        """
        turn_paths = [str(Path(dialogue_dir) / f) for f in turn_files]
        output_path = str(Path(dialogue_dir) / "dialogue_full.wav")
        duration = self.assemble(turn_paths, output_path)
        return {"full_audio": "dialogue_full.wav",
                "full_duration_sec": round(duration, 2)}
