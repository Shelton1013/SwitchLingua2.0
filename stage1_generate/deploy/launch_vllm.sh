#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — vLLM 本地模型部署脚本
#
# 服务器配置: 10× NVIDIA A6000 (48GB each), 总 VRAM 480GB
#
# 部署方案：
#   方案 A (推荐): Qwen3-235B-A22B (MoE) — tp=4, 可开 2 实例
#   方案 B: Qwen3-72B — tp=4 FP16 或 tp=1 AWQ
#   方案 C: Qwen3-32B — tp=2 FP16, 可开 4 实例最大吞吐
#
# 使用:
#   bash launch_vllm.sh [方案]
#   bash launch_vllm.sh a     # 启动 Qwen3-235B-A22B
#   bash launch_vllm.sh b     # 启动 Qwen3-72B
#   bash launch_vllm.sh c     # 启动 Qwen3-32B
# ============================================================

set -e

PLAN="${1:-a}"

# 通用参数
MAX_MODEL_LEN=4096         # CS 对话不需要长上下文
GPU_MEMORY_UTILIZATION=0.92
DTYPE="auto"               # vLLM 自动选择精度

case "$PLAN" in

  # ========================
  # 方案 A: Qwen3-235B-A22B (MoE, 推荐)
  # MoE 架构: 235B 总参数，每次推理只激活 22B
  # FP16 约 120GB VRAM → 4 张 A6000
  # 推理速度接近 32B，质量超 72B
  # 可用剩余 6 张卡开第二实例
  # ========================
  a|A)
    echo ">>> 方案 A: Qwen3-235B-A22B (MoE) — 4 GPU, FP16"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-235B-A22B \
      --tensor-parallel-size 4 \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE \
      --port 8000 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --enable-chunked-prefill \
      --max-num-seqs 64 \
      --disable-log-requests
    ;;

  # 方案 A 的第二实例（用剩余 GPU 提升吞吐）
  a2|A2)
    echo ">>> 方案 A 第二实例: Qwen3-235B-A22B — GPU 4-7, 端口 8001"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-235B-A22B \
      --tensor-parallel-size 4 \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE \
      --port 8001 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --enable-chunked-prefill \
      --max-num-seqs 64 \
      --disable-log-requests
    ;;

  # ========================
  # 方案 B: Qwen3-72B (Dense)
  # FP16 约 144GB → 4 张 A6000
  # AWQ-4bit 约 40GB → 1 张 A6000
  # ========================
  b|B)
    echo ">>> 方案 B: Qwen3-72B — 4 GPU, FP16"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-72B-Instruct \
      --tensor-parallel-size 4 \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE \
      --port 8000 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --enable-chunked-prefill \
      --max-num-seqs 128 \
      --disable-log-requests
    ;;

  # 方案 B AWQ 量化版（单卡即可，可开多实例）
  b-awq|B-AWQ)
    echo ">>> 方案 B-AWQ: Qwen3-72B-AWQ — 1 GPU, 4bit 量化"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-72B-Instruct-AWQ \
      --quantization awq \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype half \
      --port 8000 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --max-num-seqs 64 \
      --disable-log-requests
    ;;

  # ========================
  # 方案 C: Qwen3-32B (最大吞吐)
  # FP16 约 64GB → 2 张 A6000
  # 10 张卡可开 5 个实例
  # ========================
  c|C)
    echo ">>> 方案 C: Qwen3-32B — 2 GPU × 5 实例, 最大吞吐"
    echo ">>> 端口 8000-8004"

    for i in 0 1 2 3 4; do
      GPU_START=$((i * 2))
      GPU_END=$((GPU_START + 1))
      PORT=$((8000 + i))
      echo "  启动实例 $i: GPU $GPU_START,$GPU_END → 端口 $PORT"

      CUDA_VISIBLE_DEVICES=$GPU_START,$GPU_END python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-32B-Instruct \
        --tensor-parallel-size 2 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --dtype $DTYPE \
        --port $PORT \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enable-chunked-prefill \
        --max-num-seqs 128 \
        --disable-log-requests &

      sleep 5  # 等待模型加载
    done

    echo ">>> 5 个实例启动完成"
    echo ">>> 使用 dialogue_generator.py --api-base 时配合 --multi-endpoint"
    wait
    ;;

  *)
    echo "用法: bash launch_vllm.sh [a|a2|b|b-awq|c]"
    echo "  a:     Qwen3-235B-A22B (MoE, 推荐)"
    echo "  a2:    Qwen3-235B-A22B 第二实例"
    echo "  b:     Qwen3-72B FP16"
    echo "  b-awq: Qwen3-72B AWQ 量化"
    echo "  c:     Qwen3-32B × 5 实例 (最大吞吐)"
    ;;
esac
