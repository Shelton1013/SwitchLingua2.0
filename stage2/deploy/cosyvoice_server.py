"""
SwitchLingua 2.0 — CosyVoice 3 FastAPI Server

A lightweight REST API server wrapping CosyVoice for zero-shot TTS.
Place this file in the CosyVoice repo root directory, or set
COSYVOICE_ROOT env variable.

Usage:
    # From CosyVoice repo directory:
    python cosyvoice_server.py --model_dir /data/models/CosyVoice2-0.5B --port 50000

    # Or from SwitchLingua project:
    COSYVOICE_ROOT=/path/to/CosyVoice python stage2/deploy/cosyvoice_server.py \
        --model_dir /data/models/CosyVoice2-0.5B --port 50000

Endpoints:
    GET  /                          Health check
    POST /inference_zero_shot       Zero-shot voice cloning TTS
    POST /inference_instruct2       Instruction-guided TTS (with language hint)
"""

import os
import sys
import io
import wave
import struct
import argparse
import logging
import numpy as np
from pathlib import Path

# Add CosyVoice to Python path if needed
cosyvoice_root = os.environ.get("COSYVOICE_ROOT", "")
if cosyvoice_root:
    sys.path.insert(0, cosyvoice_root)
    third_party = os.path.join(cosyvoice_root, "third_party", "Matcha-TTS")
    if os.path.isdir(third_party):
        sys.path.insert(0, third_party)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import torchaudio

logger = logging.getLogger("cosyvoice_server")

app = FastAPI(title="CosyVoice TTS Server", version="1.0")

# Global model reference (loaded at startup)
cosyvoice_model = None
SAMPLE_RATE = 24000  # CosyVoice native sample rate


def load_model(model_dir: str):
    """Load CosyVoice model."""
    global cosyvoice_model

    # Try CosyVoice2 AutoModel first (recommended for CosyVoice2/3)
    try:
        from cosyvoice.cli.model import CosyVoice2Model as AutoModel
        cosyvoice_model = AutoModel(model_dir)
        logger.info(f"Loaded CosyVoice2Model from {model_dir}")
        return
    except ImportError:
        pass

    # Fallback: try CosyVoiceModel
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice
        cosyvoice_model = CosyVoice(model_dir)
        logger.info(f"Loaded CosyVoice from {model_dir}")
        return
    except ImportError:
        pass

    # Fallback: try FunASR-style AutoModel
    try:
        from model import AutoModel
        cosyvoice_model = AutoModel(model_dir)
        logger.info(f"Loaded AutoModel from {model_dir}")
        return
    except ImportError:
        raise RuntimeError(
            f"Cannot load CosyVoice model. Tried CosyVoice2Model, CosyVoice, AutoModel. "
            f"Make sure CosyVoice is installed and COSYVOICE_ROOT is set correctly."
        )


def audio_bytes_to_wav(audio_data, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert raw audio tensor/array to WAV bytes."""
    if hasattr(audio_data, 'numpy'):
        # PyTorch tensor
        audio_np = audio_data.cpu().numpy()
    elif isinstance(audio_data, np.ndarray):
        audio_np = audio_data
    else:
        audio_np = np.array(audio_data)

    # Ensure 1D
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    # Normalize to int16
    if audio_np.dtype in (np.float32, np.float64):
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np * 32767).astype(np.int16)
    elif audio_np.dtype != np.int16:
        audio_np = audio_np.astype(np.int16)

    # Write WAV
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_np.tobytes())

    return buf.getvalue()


def load_prompt_wav(file_bytes: bytes, target_sr: int = 16000):
    """Load and resample prompt WAV to target sample rate."""
    buf = io.BytesIO(file_bytes)
    speech, sr = torchaudio.load(buf)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(sr, target_sr)(speech)
    return speech


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": cosyvoice_model is not None}


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(""),
    prompt_wav: UploadFile = File(...),
):
    """Zero-shot voice cloning TTS.

    Args:
        tts_text: Text to synthesize
        prompt_text: Transcript of the reference audio
        prompt_wav: Reference audio WAV file (3-10 seconds)

    Returns:
        Streaming WAV audio
    """
    if cosyvoice_model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    try:
        # Read reference audio
        prompt_bytes = await prompt_wav.read()
        prompt_speech = load_prompt_wav(prompt_bytes)

        # Generate speech
        all_audio = []
        for result in cosyvoice_model.inference_zero_shot(
            tts_text=tts_text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech,
        ):
            if "tts_speech" in result:
                all_audio.append(result["tts_speech"])

        if not all_audio:
            return JSONResponse({"error": "No audio generated"}, status_code=500)

        # Concatenate and convert to WAV
        import torch
        full_audio = torch.cat(all_audio, dim=-1)
        wav_bytes = audio_bytes_to_wav(full_audio, SAMPLE_RATE)

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except Exception as e:
        logger.error(f"Zero-shot inference failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(...),
    instruct_text: str = Form(""),
    prompt_text: str = Form(""),
    prompt_wav: UploadFile = File(...),
):
    """Instruction-guided TTS with voice cloning.

    Args:
        tts_text: Text to synthesize
        instruct_text: Natural language instruction (e.g. "请用粤语朗读")
        prompt_text: Transcript of the reference audio
        prompt_wav: Reference audio WAV file

    Returns:
        Streaming WAV audio
    """
    if cosyvoice_model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    try:
        prompt_bytes = await prompt_wav.read()
        prompt_speech = load_prompt_wav(prompt_bytes)

        all_audio = []

        # Try instruct2 method first (CosyVoice 3)
        if hasattr(cosyvoice_model, 'inference_instruct2'):
            for result in cosyvoice_model.inference_instruct2(
                tts_text=tts_text,
                instruct_text=instruct_text,
                prompt_speech_16k=prompt_speech,
            ):
                if "tts_speech" in result:
                    all_audio.append(result["tts_speech"])

        # Fallback: try instruct method (CosyVoice 1/2)
        elif hasattr(cosyvoice_model, 'inference_instruct'):
            for result in cosyvoice_model.inference_instruct(
                tts_text=tts_text,
                spk_id="",
                instruct_text=instruct_text,
            ):
                if "tts_speech" in result:
                    all_audio.append(result["tts_speech"])

        # Final fallback: use zero_shot (ignore instruct)
        else:
            logger.warning(
                "Model has no instruct method, falling back to zero_shot"
            )
            for result in cosyvoice_model.inference_zero_shot(
                tts_text=tts_text,
                prompt_text=prompt_text or instruct_text,
                prompt_speech_16k=prompt_speech,
            ):
                if "tts_speech" in result:
                    all_audio.append(result["tts_speech"])

        if not all_audio:
            return JSONResponse({"error": "No audio generated"}, status_code=500)

        import torch
        full_audio = torch.cat(all_audio, dim=-1)
        wav_bytes = audio_bytes_to_wav(full_audio, SAMPLE_RATE)

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except Exception as e:
        logger.error(f"Instruct2 inference failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--model_dir", required=True, help="Path to CosyVoice model")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(f"Loading model from {args.model_dir}...")
    load_model(args.model_dir)
    logger.info(f"Model loaded. Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
