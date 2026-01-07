#!/bin/bash
set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image_name="vllm-nvfp4:latest"
vllm_wheel="vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl"
vllm_version="v0.13.0"

function download_vllm() {
  local wheel_path="$script_dir/$vllm_wheel"
  if [ -f "$wheel_path" ]; then
    echo "File '$vllm_wheel' already exists. Skipping download."
  else
    echo "Downloading vLLM $vllm_version wheel..."
    wget "https://github.com/vllm-project/vllm/releases/download/$vllm_version/$vllm_wheel" \
      -O "$wheel_path"
  fi
}

function build_docker() {
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

echo "Please be patient, this takes 8-10+ minutes. Building vLLM $vllm_version container..."
build_docker
validate_image