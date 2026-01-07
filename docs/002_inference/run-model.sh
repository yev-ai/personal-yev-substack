#!/bin/bash
set -e

model="nvidia/Qwen3-30B-A3B-NVFP4"
image="vllm-nvfp4:latest"
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

function run_docker() {
  docker run --name vllm-nvfp4 \
    --rm \
    --pid=host \
    --ipc=host \
    --gpus all \
    --runtime nvidia \
    -v "$HF_HOME":/root/.cache/huggingface \
    -p $port:8000 \
    "$image" \
    --model "$model" \
    --trust-remote-code \
    --no-enable-chunked-prefill \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
}

validate_hf_home

echo "Searching for $model in $HF_HOME..."
validate_model

echo "Starting vLLM container..."
run_docker
