#!/bin/bash
# ============================================================
# SwitchLingua 2.0 — vLLM 本地模型部署脚本
#
# 服务器配置: 10× NVIDIA A6000 (48GB each), 总 VRAM 480GB
#
# 部署方案：
#   方案 A (推荐): Qwen3-235B-A22B (MoE) — tp=4, 可开 2 实例
#   方案 B: Qwen3-72B — tp=4 FP16 或 tp=1 AWQ
#   方案 C: Qwen3-32B — tp=2 FP16, 可开 5 实例最大吞吐
#
# 使用:
#   bash launch_vllm.sh [方案] [模型路径]
#
#   # 从 HuggingFace 自动下载（需要网络）
#   bash launch_vllm.sh a
#
#   # 使用本地已下载的模型（推荐）
#   bash launch_vllm.sh a /data/models/Qwen3-235B-A22B
#   bash launch_vllm.sh b /data/models/Qwen3-72B-Instruct
#   bash launch_vllm.sh c /data/models/Qwen3-32B-Instruct
#
# 模型下载方式（提前在有网络的环境下载好）：
#   # 方式 1: HuggingFace 镜像（国内推荐）
#   export HF_ENDPOINT=https://hf-mirror.com
#   huggingface-cli download Qwen/Qwen3-235B-A22B \
#       --local-dir /data/models/Qwen3-235B-A22B
#
#   # 方式 2: ModelScope（阿里云，国内最快）
#   pip install modelscope
#   modelscope download Qwen/Qwen3-235B-A22B \
#       --local_dir /data/models/Qwen3-235B-A22B
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
    # 用户指定了本地路径
    if [ ! -d "$MODEL_PATH" ]; then
      echo "错误: 模型路径不存在: $MODEL_PATH" >&2
      exit 1
    fi
    echo "$MODEL_PATH"
  else
    # 使用 HuggingFace ID（自动下载）
    echo "$default_hf_id"
  fi
}

case "$PLAN" in

  # ========================
  # 方案 A: Qwen3-235B-A22B (MoE, 推荐)
  # MoE 架构: 235B 总参数，每次推理只激活 22B
  # FP16 约 120GB VRAM → 4 张 A6000
  # 推理速度接近 32B，质量超 72B
  # 可用剩余 6 张卡开第二实例
  # ========================
  a|A)
    MODEL=$(get_model "Qwen/Qwen3-235B-A22B")
    echo ">>> 方案 A: Qwen3-235B-A22B (MoE) — 4 GPU, FP16"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
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
    MODEL=$(get_model "Qwen/Qwen3-235B-A22B")
    echo ">>> 方案 A 第二实例: Qwen3-235B-A22B — GPU 4-7, 端口 8001"
    echo ">>> 模型路径: $MODEL"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
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
  # ========================
  b|B)
    MODEL=$(get_model "Qwen/Qwen3-72B-Instruct")
    echo ">>> 方案 B: Qwen3-72B — 4 GPU, FP16"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
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
    MODEL=$(get_model "Qwen/Qwen3-72B-Instruct-AWQ")
    echo ">>> 方案 B-AWQ: Qwen3-72B-AWQ — 1 GPU, 4bit 量化"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000"

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
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
    MODEL=$(get_model "Qwen/Qwen3-32B-Instruct")
    echo ">>> 方案 C: Qwen3-32B — 2 GPU × 5 实例, 最大吞吐"
    echo ">>> 模型路径: $MODEL"
    echo ">>> 端口 8000-8004"

    for i in 0 1 2 3 4; do
      GPU_START=$((i * 2))
      GPU_END=$((GPU_START + 1))
      PORT=$((8000 + i))
      echo "  启动实例 $i: GPU $GPU_START,$GPU_END → 端口 $PORT"

      CUDA_VISIBLE_DEVICES=$GPU_START,$GPU_END python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
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
    echo ">>> 使用 --api-base http://localhost:8000/v1 ... http://localhost:8004/v1"
    wait
    ;;

  # ========================
  # download: 仅下载模型，不启动服务
  # ========================
  download|dl)
    HF_ID="${MODEL_PATH:-Qwen/Qwen3-235B-A22B}"
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
    echo "  a          Qwen3-235B-A22B (MoE, 推荐)"
    echo "  a2         Qwen3-235B-A22B 第二实例 (GPU 4-7)"
    echo "  b          Qwen3-72B FP16"
    echo "  b-awq      Qwen3-72B AWQ 量化 (单卡)"
    echo "  c          Qwen3-32B × 5 实例 (最大吞吐)"
    echo "  download   仅下载模型 (不启动服务)"
    echo ""
    echo "示例:"
    echo "  # 自动从 HuggingFace 下载"
    echo "  bash launch_vllm.sh a"
    echo ""
    echo "  # 先下载到本地"
    echo "  bash launch_vllm.sh download Qwen/Qwen3-235B-A22B /data/models/Qwen3-235B-A22B"
    echo ""
    echo "  # 用本地模型启动"
    echo "  bash launch_vllm.sh a /data/models/Qwen3-235B-A22B"
    echo "  bash launch_vllm.sh a2 /data/models/Qwen3-235B-A22B"
    ;;
esac
