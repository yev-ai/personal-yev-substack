#!/bin/bash
set -e

model="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"
vllm_cache_dir="$HOME/.cache/vllm"
image="vllm-nightly:latest"
port=1337

function validate_hf_home() {
  if [ -z "$HF_HOME" ]; then
    echo "Error: HF_HOME environment variable is not set."
    exit 1
  fi
  
  if [ ! -d "$HF_HOME" ]; then
    echo "Error: The directory specified in HF_HOME ($HF_HOME) does not exist."
    exit 1
  fi
}

function validate_model() {
  local model_folder="models--${model//\//--}"
  FOUND_PATH=$(find "$HF_HOME" -type d -name "$model_folder" -print -quit)
  
  if [ -z "$FOUND_PATH" ]; then
    echo "Error: Could not find the model folder '$model_folder' inside $HF_HOME."
    echo " Have you downloaded it with: hf download $model?"
    exit 1
  fi
  
  echo "✅ Found model at: $FOUND_PATH"
}

function setup_fs() {
  if [ -d "$vllm_cache_dir" ]; then
    echo "vLLM cache folder already exists at: $vllm_cache_dir"
  else
    echo "Creating vLLM cache folder at: $vllm_cache_dir"
    mkdir -p "$vllm_cache_dir"
  fi
}

function run_docker() {
  docker run --name vllm-nvfp4-dev \
    --rm \
    --pid=host \
    --ipc=host \
    --gpus all \
    --runtime nvidia \
    -v "$HF_HOME":/root/.cache/huggingface \
    -v "$vllm_cache_dir":/root/.cache/vllm \
    -p $port:8000 \
    -e VLLM_USE_V1=1 \
    -e VLLM_FLASH_ATTN_VERSION=2 \
    -e TORCH_CUDA_ARCH_LIST="12.0" \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e TORCH_CUDNN_V8_API_ENABLED=1 \
    -e HF_TOKEN="$HF_TOKEN" \
    "$image" \
    --model "$model" \
    --quantization modelopt_fp4 \
    --attention-backend flashinfer \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cache_dir": "/root/.cache/vllm"}' \
    --max_model_len 131072 \
    --max-num-batched-tokens 65536 \
    --kv-cache-dtype fp8_e4m3 \
    --max-num-seqs 1 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.88
}

validate_hf_home

echo "Searching for $model in $HF_HOME..."
validate_model

setup_fs

echo "Starting vLLM container..."
run_docker
