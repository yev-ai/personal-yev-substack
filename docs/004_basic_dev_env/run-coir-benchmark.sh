#!/bin/bash
set -e

IMAGE_NAME="coir-benchmark-runner"
RESULTS_DIR="$(pwd)/benchmark_results"
CACHE_DIR="$(pwd)/cache"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CACHE_DIR"

echo "🔨 Building Image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" -f Dockerfile-benchmark .

echo "🚀 Starting Benchmark..."
echo "   - Results: $RESULTS_DIR"

docker run --rm \
  --network host \
  -v "$RESULTS_DIR":/app/results \
  -v "$CACHE_DIR":/root/.cache \
  "$IMAGE_NAME"