#!/bin/bash
set -e

function run_manager() {
  echo "Starting Embedding Manager..."
  docker run --rm \
    --name embedding-manager \
    --gpus all \
    --runtime nvidia \
    --pid=host \
    --ipc=host \
    --network host \
    -e TEI_BASE_URL="http://127.0.0.1:1336" \
    -e PORT=1335 \
    embedding-manager
}

run_manager
