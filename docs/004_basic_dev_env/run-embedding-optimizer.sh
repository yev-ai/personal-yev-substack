#!/bin/bash
set -e

function run_proxy() {
  echo "Starting Embedding Optimizer..."
  docker run --rm \
    --name embedding-optimizer \
    --network host \
    -e TEI_BASE_URL="http://127.0.0.1:1336" \
    -e PORT=1335 \
    embedding-optimizer
}

run_proxy