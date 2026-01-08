#!/bin/bash
set -e

# Config
MODEL="jinaai/jina-code-embeddings-0.5b"
MODEL_FOLDER="models--${MODEL//\//--}"
IMAGE="embedding-inference:latest"
PORT=1336

function validate_hf_home() {
  if [ -z "$HF_HOME" ]; then
    echo "Error: HF_HOME environment variable is not set."
    exit 1
  fi
}

function get_snapshot_dir() {
  local model_path="$HF_HOME/hub/$MODEL_FOLDER"
  
  if [ ! -d "$model_path" ]; then
    echo "Error: Model not found. Run 'hf download $MODEL'"
    exit 1
  fi

  SNAPSHOT_DIR=$(find "$model_path/snapshots" -maxdepth 1 -mindepth 1 -type d | head -n 1)
  
  if [ -z "$SNAPSHOT_DIR" ]; then
    echo "Error: No snapshot found in $model_path/snapshots"
    exit 1
  fi
}

function ensure_tokenizer() {
  if [ -f "$SNAPSHOT_DIR/tokenizer.json" ]; then
    echo "tokenizer.json exists"
    return 0
  fi

  echo "Generating tokenizer.json (this may take 30-90 seconds)..."
  docker run --rm \
    -v "$HF_HOME":/hf \
    -e HF_HOME=/hf \
    -e HF_HOME_HOST="$HF_HOME" \
    -e MODEL_ID="$MODEL" \
    -e MODEL_FOLDER="$MODEL_FOLDER" \
    python:3.12-slim \
    sh -c 'pip install -q --root-user-action=ignore transformers && python3 << "PYEOF"
import os
from transformers import AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
HF_HOME = os.environ["HF_HOME"]
HF_HOME_HOST = os.environ.get("HF_HOME_HOST", HF_HOME)

model_folder = os.environ.get("MODEL_FOLDER")
search_path = os.path.join(HF_HOME, "hub", model_folder, "snapshots")

snapshots = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
if not snapshots:
    raise SystemExit(f"No snapshots found in {search_path}")

latest_snapshot = os.path.join(search_path, snapshots[0])

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Saving to {latest_snapshot}...")
tokenizer.save_pretrained(latest_snapshot)

target_file = os.path.join(latest_snapshot, "tokenizer.json")

if os.path.exists(target_file):
    host_path = target_file.replace("/hf", HF_HOME_HOST)
    print(f"Success! Created: {host_path}")
else:
    raise SystemExit("Failed to create tokenizer.json")
PYEOF'
  echo "tokenizer.json generated"
}

function run_tei() {
  local search_path="$HF_HOME/hub"
  local relative_path="${SNAPSHOT_DIR#$search_path/}"
  local container_model_path="/data/$relative_path"

  echo "Host Hub Path:      $search_path"
  echo "Model Snapshot:     $relative_path"
  echo "Container Path:     $container_model_path"

  if [ "${BENCHMARK}" = "true" ]; then
    local max_client_batch_size=1024
    local max_batch_tokens=252144
    echo "Running in BENCHMARK mode (take all the VRAM)"
  else
    # These settings cap it to 5-7% GPU mem + core, gracefully sharing resources with
    # NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4 set at 0.88 mem, just near the safety cap
    # Note: this is tuned *VERY* specifically for the hardware in Chapter 1 on Substack
    local max_client_batch_size=64
    local max_batch_tokens=32768
    echo "Running in DEVELOPMENT mode (we chose peace)"
  fi

  docker run --rm \
    --name embedding-inference \
    --gpus all \
    --runtime nvidia \
    -p $PORT:80 \
    -e HF_HUB_OFFLINE=1 \
    --user $(id -u):$(id -g) \
    -v "$search_path":/data \
    "$IMAGE" \
    --model-id "$container_model_path" \
    --pooling last-token \
    --max-client-batch-size "$max_client_batch_size" \
    --max-batch-tokens "$max_batch_tokens"
}

validate_hf_home
get_snapshot_dir
ensure_tokenizer

echo "Starting TEI container (Blackwell Native)..."
run_tei
