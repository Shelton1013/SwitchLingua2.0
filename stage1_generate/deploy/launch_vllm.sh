#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — vLLM 本地模型部署脚本
#
# 服务器配置: 10× NVIDIA A6000 (48GB each), 总 VRAM 480GB
# 模型: Qwen3.5 系列 (MoE 多模态，支持纯文本生成)
#
# 部署方案：
#   方案 A (推荐): Qwen3.5-122B-A10B-FP8 — tp=3, 可开 3 实例
#   方案 B: Qwen3.5-397B-A17B-GPTQ-Int4 — tp=5, 最强质量
#   方案 C: Qwen3.5-35B-A3B — tp=1, 可开 10 实例最大吞吐
#
# 使用:
#   bash launch_vllm.sh [方案] [模型路径]
#
#   # 使用本地已下载的模型（推荐）
#   bash launch_vllm.sh a /data/models/Qwen3.5-122B-A10B-FP8
#
#   # 从 HuggingFace 自动下载（需要网络）
#   bash launch_vllm.sh a
#
# 模型下载方式（提前在有网络的环境下载好）：
#   # 方式 1: HuggingFace 镜像（国内推荐）
#   export HF_ENDPOINT=https://hf-mirror.com
#   huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8 \
#       --local-dir /data/models/Qwen3.5-122B-A10B-FP8
#
#   # 方式 2: ModelScope（阿里云，国内最快）
#   pip install modelscope
#   modelscope download Qwen/Qwen3.5-122B-A10B-FP8 \
#       --local_dir /data/models/Qwen3.5-122B-A10B-FP8
# ============================================================

set -e

PLAN="${1:-a}"
MODEL_PATH="${2:-}"  # 第二个参数：本地模型路径（可选）

# 通用参数
MAX_MODEL_LEN=4096         # CS 对话不需要长上下文
GPU_MEMORY_UTILIZATION=0.92
DTYPE="auto"               # vLLM 自动选择精度

# ============================================================
# 根据方案确定默认模型名（未指定本地路径时使用 HF ID）
# ============================================================
get_model() {
  local default_hf_id="$1"
  if [ -n "$MODEL_PATH" ]; then
    if [ ! -d "$MODEL_PATH" ]; then
      echo "错误: 模型路径不存在: $MODEL_PATH" >&2
      exit 1
    fi
    echo "$MODEL_PATH"
  else
    echo "$default_hf_id"
  fi
}

case "$PLAN" in

  # ========================
  # 方案 A: Qwen3.5-122B-A10B-FP8 (推荐)
  # MoE: 122B 总参数，10B 激活，FP8 量化
  # ~125GB VRAM → 3 张 A6000
  # 10 张卡可开 3 实例 (3+3+3=9 GPU, 1 空闲)
  # 推理快，质量强于 Qwen3-235B
  # ========================
  a|A)
    MODEL=$(get_model "Qwen/Qwen3.5-122B-A10B-FP8")
    echo ">>> 方案 A: Qwen3.5-122B-A10B-FP8 (MoE) — 3 GPU, FP8"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0,1,2 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 3 \
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

  # 方案 A 第二实例
  a2|A2)
    MODEL=$(get_model "Qwen/Qwen3.5-122B-A10B-FP8")
    echo ">>> 方案 A 第二实例 — GPU 3-5, 端口 8001"
    echo ">>> 模型路径: $MODEL"

    CUDA_VISIBLE_DEVICES=3,4,5 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 3 \
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

  # 方案 A 第三实例
  a3|A3)
    MODEL=$(get_model "Qwen/Qwen3.5-122B-A10B-FP8")
    echo ">>> 方案 A 第三实例 — GPU 6-8, 端口 8002"
    echo ">>> 模型路径: $MODEL"

    CUDA_VISIBLE_DEVICES=6,7,8 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 3 \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE \
      --port 8002 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --enable-chunked-prefill \
      --max-num-seqs 64 \
      --disable-log-requests
    ;;

  # 方案 A 全部 3 实例一键启动
  a-all|A-ALL)
    MODEL=$(get_model "Qwen/Qwen3.5-122B-A10B-FP8")
    echo ">>> 方案 A: 3 实例一键启动"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000, 8001, 8002"

    CUDA_VISIBLE_DEVICES=0,1,2 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --tensor-parallel-size 3 \
      --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE --port 8000 --host 0.0.0.0 --trust-remote-code \
      --enable-chunked-prefill --max-num-seqs 64 --disable-log-requests &

    sleep 10

    CUDA_VISIBLE_DEVICES=3,4,5 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --tensor-parallel-size 3 \
      --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE --port 8001 --host 0.0.0.0 --trust-remote-code \
      --enable-chunked-prefill --max-num-seqs 64 --disable-log-requests &

    sleep 10

    CUDA_VISIBLE_DEVICES=6,7,8 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --tensor-parallel-size 3 \
      --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype $DTYPE --port 8002 --host 0.0.0.0 --trust-remote-code \
      --enable-chunked-prefill --max-num-seqs 64 --disable-log-requests &

    echo ">>> 3 实例启动中... 请等待模型加载完成"
    echo ">>> 生成命令: python dialogue_generator.py --api-base http://localhost:8000/v1 http://localhost:8001/v1 http://localhost:8002/v1"
    wait
    ;;

  # ========================
  # 方案 B: Qwen3.5-397B-A17B-GPTQ-Int4 (最强质量)
  # MoE: 397B 总参数，17B 激活，GPTQ-4bit
  # ~200GB VRAM → 5 张 A6000
  # 可开 2 实例 (5+5=10 GPU)
  # ========================
  b|B)
    MODEL=$(get_model "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")
    echo ">>> 方案 B: Qwen3.5-397B-A17B-GPTQ-Int4 — 5 GPU, 最强质量"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 5 \
      --quantization gptq \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype half \
      --port 8000 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --enable-chunked-prefill \
      --max-num-seqs 32 \
      --disable-log-requests
    ;;

  # 方案 B 第二实例
  b2|B2)
    MODEL=$(get_model "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")
    echo ">>> 方案 B 第二实例 — GPU 5-9, 端口 8001"
    echo ">>> 模型路径: $MODEL"

    CUDA_VISIBLE_DEVICES=5,6,7,8,9 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 5 \
      --quantization gptq \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --dtype half \
      --port 8001 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --enable-chunked-prefill \
      --max-num-seqs 32 \
      --disable-log-requests
    ;;

  # ========================
  # 方案 C: Qwen3.5-35B-A3B (最大吞吐)
  # MoE: 35B 总参数，3B 激活
  # FP16 ~20GB → 1 张 A6000
  # 10 张卡可开 10 个实例
  # ========================
  c|C)
    MODEL=$(get_model "Qwen/Qwen3.5-35B-A3B")
    echo ">>> 方案 C: Qwen3.5-35B-A3B — 1 GPU × 10 实例, 最大吞吐"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000-8009"

    for i in $(seq 0 9); do
      PORT=$((8000 + i))
      echo "  启动实例 $i: GPU $i → 端口 $PORT"

      CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --dtype $DTYPE \
        --port $PORT \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enable-chunked-prefill \
        --max-num-seqs 128 \
        --disable-log-requests &

      sleep 3
    done

    echo ">>> 10 个实例启动完成"
    echo ">>> 生成命令: python dialogue_generator.py --api-base http://localhost:800{0..9}/v1"
    wait
    ;;

  # ========================
  # download: 仅下载模型，不启动服务
  # ========================
  download|dl)
    HF_ID="${MODEL_PATH:-Qwen/Qwen3.5-122B-A10B-FP8}"
    LOCAL_DIR="${3:-/data/models/$(basename $HF_ID)}"
    echo ">>> 下载模型: $HF_ID → $LOCAL_DIR"

    # 优先使用镜像
    if [ -z "$HF_ENDPOINT" ]; then
      export HF_ENDPOINT=https://hf-mirror.com
      echo ">>> 使用 HF 镜像: $HF_ENDPOINT"
    fi

    huggingface-cli download "$HF_ID" --local-dir "$LOCAL_DIR"
    echo ">>> 下载完成: $LOCAL_DIR"
    echo ">>> 启动命令: bash launch_vllm.sh a $LOCAL_DIR"
    ;;

  *)
    echo "用法: bash launch_vllm.sh [方案] [本地模型路径]"
    echo ""
    echo "方案:"
    echo "  a          Qwen3.5-122B-A10B-FP8 (推荐, 3 GPU)"
    echo "  a2         第二实例 (GPU 3-5)"
    echo "  a3         第三实例 (GPU 6-8)"
    echo "  a-all      3 实例一键启动 (GPU 0-8)"
    echo "  b          Qwen3.5-397B-A17B-GPTQ-Int4 (最强, 5 GPU)"
    echo "  b2         第二实例 (GPU 5-9)"
    echo "  c          Qwen3.5-35B-A3B × 10 实例 (最大吞吐)"
    echo "  download   仅下载模型"
    echo ""
    echo "示例:"
    echo "  # 下载模型"
    echo "  bash launch_vllm.sh download Qwen/Qwen3.5-122B-A10B-FP8 /data/models/Qwen3.5-122B-A10B-FP8"
    echo ""
    echo "  # 用本地模型启动"
    echo "  bash launch_vllm.sh a /data/models/Qwen3.5-122B-A10B-FP8"
    echo ""
    echo "  # 3 实例一键启动（推荐）"
    echo "  bash launch_vllm.sh a-all /data/models/Qwen3.5-122B-A10B-FP8"
    ;;
esac
