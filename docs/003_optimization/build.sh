#!/bin/bash
set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image_name="vllm-nightly:latest"

# 0.14.0rc1 wheel built from https://github.com/vllm-project/vllm/tree/c7a79d41a03f925942e8fb8bc589df4f39bcb950
wheel_url="https://wheels.vllm.ai/c7a79d41a03f925942e8fb8bc589df4f39bcb950/vllm-0.14.0rc1.dev323%2Bgc7a79d41a-cp38-abi3-manylinux_2_31_x86_64.whl"
vllm_wheel="vllm-0.14.0rc1.dev323+gc7a79d41a-cp38-abi3-manylinux_2_31_x86_64.whl"

function download_vllm() {
  local wheel_path="$script_dir/$vllm_wheel"
  
  if [ -f "$wheel_path" ]; then
    echo "File '$vllm_wheel' already exists. Skipping download."
  else
    echo "Downloading vLLM Nightly ($vllm_wheel)..."
    wget "$wheel_url" -O "$wheel_path"
  fi
}

function build_docker() {
  echo "Building Docker image '$image_name' using wheel: $vllm_wheel"
  
  docker build -t $image_name \
    --build-arg VLLM_WHEEL="$vllm_wheel" \
    "$script_dir"
}

function validate_image() {
  if [[ "$(docker images -q "$image_name" 2> /dev/null)" == "" ]]; then
    echo "Error: Docker image '$image_name' was not found."
    exit 1
  else
    echo "Success! Image '$image_name' verified."
  fi
}

download_vllm

echo "Please be patient, this takes 8-10+ minutes..."
build_docker
validate_image