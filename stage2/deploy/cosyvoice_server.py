"""
SwitchLingua 2.0 — CosyVoice TTS Server

Based on the official CosyVoice runtime/python/fastapi/server.py,
with added vLLM acceleration support.

Usage:
    # Basic (no vLLM)
    python cosyvoice_server.py --model_dir /data/models/Fun-CosyVoice3-0.5B-2512

    # With vLLM acceleration
    python cosyvoice_server.py --model_dir /data/models/Fun-CosyVoice3-0.5B-2512 --vllm
"""

import argparse
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn

logger = logging.getLogger("cosyvoice_server")

app = FastAPI(title="CosyVoice TTS Server")

# Global model reference
cosyvoice = None


def generate_audio(model_output):
    """Yield raw int16 PCM bytes from model output generator."""
    for result in model_output:
        tts_audio = result["tts_speech"].numpy().flatten()
        yield (tts_audio * (2 ** 15)).astype(np.int16).tobytes()


@app.get("/")
async def health():
    return {"status": "ok", "model_loaded": cosyvoice is not None}


PROMPT_MARKER = "<|endofprompt|>"


def ensure_prompt_marker(prompt_text: str) -> str:
    """CosyVoice 3 requires <|endofprompt|> in prompt_text."""
    if PROMPT_MARKER not in prompt_text:
        return f"You are a helpful assistant.{PROMPT_MARKER}{prompt_text}"
    return prompt_text


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File(),
):
    """Zero-shot voice cloning TTS."""
    prompt_wav_path = save_wav_temp(await prompt_wav.read())
    model_output = cosyvoice.inference_zero_shot(
        tts_text, ensure_prompt_marker(prompt_text),
        prompt_wav_path, stream=False,
    )
    return StreamingResponse(generate_audio(model_output))


@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(),
    instruct_text: str = Form(""),
    prompt_text: str = Form(""),
    prompt_wav: UploadFile = File(),
):
    """Instruction-guided TTS with voice cloning."""
    prompt_wav_path = save_wav_temp(await prompt_wav.read())

    # Try instruct2 first, fallback to zero_shot
    if hasattr(cosyvoice, "inference_instruct2"):
        model_output = cosyvoice.inference_instruct2(
            tts_text, instruct_text, prompt_wav_path, stream=False,
        )
    else:
        logger.warning("inference_instruct2 not available, falling back to zero_shot")
        model_output = cosyvoice.inference_zero_shot(
            tts_text, prompt_text or instruct_text, prompt_wav_path, stream=False,
        )
    return StreamingResponse(generate_audio(model_output))


@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(),
    prompt_wav: UploadFile = File(),
):
    """Cross-lingual TTS."""
    prompt_wav_path = save_wav_temp(await prompt_wav.read())
    model_output = cosyvoice.inference_cross_lingual(
        tts_text, prompt_wav_path, stream=False,
    )
    return StreamingResponse(generate_audio(model_output))


def save_wav_temp(file_bytes: bytes) -> str:
    """Save uploaded WAV bytes to a temp file and return the path.

    CosyVoice's inference methods expect a file path string, not a tensor.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(file_bytes)
    tmp.close()
    return tmp.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Local path to CosyVoice model")
    parser.add_argument("--vllm", action="store_true",
                        help="Enable vLLM acceleration")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 inference")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Register vLLM model if needed
    if args.vllm:
        logger.info("Registering CosyVoice vLLM model...")
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    # Load model
    logger.info(f"Loading model from {args.model_dir} (vllm={args.vllm})...")
    from cosyvoice.cli.cosyvoice import AutoModel

    model_kwargs = {"model_dir": args.model_dir}
    if args.vllm:
        model_kwargs["load_vllm"] = True
        model_kwargs["load_trt"] = True
    if args.fp16:
        model_kwargs["fp16"] = True

    cosyvoice = AutoModel(**model_kwargs)
    logger.info(f"Model loaded: {type(cosyvoice).__name__}, sample_rate={cosyvoice.sample_rate}")

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
