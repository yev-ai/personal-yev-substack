#!/bin/bash
set -e

HF_INFO=$(hf env)
HF_MODELS=$(hf cache ls)
HF_HUB_CACHE=$(echo "$HF_INFO" | grep "HF_HUB_CACHE:" | awk -F: '{print $2}' | xargs)

echo $HF_HUB_CACHE

# Models to check
REQUIRED_MODELS=(
  "model/NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"
  "model/jinaai/jina-code-embeddings-0.5b"
)

function ensure_required_models() {

  for model in "${REQUIRED_MODELS[@]}"; do
    if echo "$HF_MODELS" | grep -q "$model"; then
      echo "✓ Found: $model"
    else
      echo "✗ Missing: $model"
      exit 1
    fi
  done

  echo "All required models are available."
}

function get_snapshot_dir() {
  local model_name="$1"
  local model_name_clean="${model_name#model/}"
  local model_path="$HF_HUB_CACHE/models--${model_name_clean//\//--}"

  if [ ! -d "$model_path" ]; then
    echo "Error: Model not found at $model_path"
    exit 1
  fi

  SNAPSHOT_DIR=$(find "$model_path/snapshots" -maxdepth 1 -mindepth 1 -type d | head -n 1)

  if [ -z "$SNAPSHOT_DIR" ]; then
    echo "Error: No snapshot found in $model_path/snapshots"
    exit 1
  fi

  local env_key="${model_name_clean#*/}"
  env_key="${env_key//-/_}"
  echo "$env_key=\"$SNAPSHOT_DIR\"" >> .env
}

for model in "${REQUIRED_MODELS[@]}"; do
  get_snapshot_dir "$model"
done
