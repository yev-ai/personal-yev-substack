#!/bin/bash
set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

function setup_fs() {
  local data_dir="$script_dir/webui_data"
  if [ -d "$data_dir" ]; then
    echo "WebUI data folder already exists at: $data_dir"
  else
    echo "Creating WebUI data folder at: $data_dir"
    mkdir -p "$data_dir"
  fi
}

function run_docker() {
  docker run --name vllm-ui \
    --rm \
    --network=host \
    -e PORT=1338 \
    -v "$script_dir/webui_data":/app/backend/data \
    -e OPENAI_API_BASE_URL=http://localhost:1337/v1 \
    -e OPENAI_API_KEY=empty \
    -e WEBUI_AUTH=False \
    -e ENABLE_OLLAMA_API="False" \
    -e RAG_EMBEDDING_MODEL="" \
    -e RAG_RERANKING_MODEL="" \
    ghcr.io/open-webui/open-webui:main
}

setup_fs

echo "Starting Open WebUI..."
run_docker