"""
Fish Speech S2 单条测试脚本

用法：
    # 1. 先启动 Fish Speech 服务
    # 2. 准备一段参考音频（3-10秒）
    # 3. 运行：

    python stage2/test_fish_speech.py \
        --text "喂，周末有冇計劃去搵啲好食嘅地方？我最近諗住試下新開嗰間restaurant。" \
        --ref-audio /path/to/reference.wav \
        --ref-text "参考音频里说的话" \
        --output /tmp/test_output.wav \
        --fish-url http://localhost:8080
"""

import argparse
import requests
import sys


def synthesize(text, ref_audio_path, ref_text, fish_url, output_path):
    """单条文本合成"""

    # 读取参考音频
    with open(ref_audio_path, "rb") as f:
        ref_audio_bytes = f.read()

    print(f"Text:      {text}")
    print(f"Ref audio: {ref_audio_path}")
    print(f"Ref text:  {ref_text}")
    print(f"Server:    {fish_url}")
    print()

    # 尝试用 ormsgpack（支持二进制音频内联）
    try:
        import ormsgpack

        request_data = {
            "text": text,
            "references": [
                {
                    "audio": ref_audio_bytes,
                    "text": ref_text,
                }
            ],
            "format": "wav",
            "temperature": 0.8,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "normalize": True,
        }

        print("Sending request (msgpack)...")
        resp = requests.post(
            f"{fish_url}/v1/tts",
            data=ormsgpack.packb(request_data),
            headers={"Content-Type": "application/msgpack"},
            timeout=120,
        )
    except ImportError:
        print("ormsgpack not installed, using JSON (no ref audio inline)")
        resp = requests.post(
            f"{fish_url}/v1/tts",
            json={"text": text, "format": "wav"},
            timeout=120,
        )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)

    wav_bytes = resp.content
    with open(output_path, "wb") as f:
        f.write(wav_bytes)

    # 计算时长
    import wave, io
    try:
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
        print(f"OK! Saved to {output_path} ({duration:.2f}s, {len(wav_bytes)} bytes)")
    except Exception:
        print(f"OK! Saved to {output_path} ({len(wav_bytes)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Speech S2 单条测试")
    parser.add_argument("--text", required=True, help="要合成的文本")
    parser.add_argument("--ref-audio", required=True, help="参考音频路径 (3-10s WAV/MP3)")
    parser.add_argument("--ref-text", default="", help="参考音频的文字转录")
    parser.add_argument("--output", default="/tmp/fish_test.wav", help="输出路径")
    parser.add_argument("--fish-url", default="http://localhost:8080", help="Fish Speech API URL")
    args = parser.parse_args()

    # 健康检查
    try:
        r = requests.get(f"{args.fish_url}/v1/health", timeout=5)
        if r.status_code != 200:
            print(f"Server not healthy: {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot connect to {args.fish_url}: {e}")
        sys.exit(1)

    synthesize(args.text, args.ref_audio, args.ref_text, args.fish_url, args.output)
