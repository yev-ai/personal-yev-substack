[WIP - WORK IN PROGRESS]

# Docker Installation

We should **install Docker Engine inside WSL** so that `nvidia-container-toolkit` can interface directly with the WSL Linux kernel via [dxgkrnl](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/directx-graphics-kernel-subsystem). This lets WSL  sub in for a native host OS by directly managing the GPU resources.

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Add user to group
sudo usermod -aG docker $USER && newgrp docker

# Confirm socket (should show CONTAINER ID in red)
docker ps | grep "CONTAINER ID"
```

# HuggingFace!

```bash
# Install the HuggingFace CLI
brew install huggingface-cli

# Enable Git credential store
git config --global credential.helper store

# Log In (also store as git credential)
hf auth login

# Confirm you're logged in
hf auth whoami

# Create a folder for HuggingFace models
mkdir "$HOME/HF_Models"

# Add it to the pre-zshrc hook from Chapter 2
echo 'export HF_HOME="$HOME/HF_Models"' >> "$HOME/.zshrc-pre.sh" && source "$HOME/.zshrc"

# Confirm it's available
echo $HF_HOME
```

The casual favorite. Go ahead and make an account on HuggingFace and go here to create a fine-grained access token.

# vLLM Installation

Blackwell GPUs support [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (1 sign, 2 exponent, and 1 mantissa bit) models and we're going to be heavily abusing this. This groups weights into blocks of 16, each of which shares a high-precision 8-bit (E4M3) scale factor and allows the 4-bit quantized coefficients to handle the finer details.

They also come with the [2nd Gen Transformer Engine](https://github.com/NVIDIA/TransformerEngine), which which has Tensor Core instructions that operate directly on the compressed 4-bit blocks. The result of this is much more efficient use of VRAM that lets us run much larger models with only ~28GB of available VRAM (4GB for host OS).

[vLLM](https://github.com/vllm-project/vllm) recently added experimental support for NVFP4 so we don't have to build [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) from source and [compile our models for TRT-LLM](https://github.com/NVIDIA/Model-Optimizer) to get reasonable [TTFT (time to first token) and TPS (tokens per second)](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html). vLLM also comes with a number of optimizations like [PagedAttention](https://arxiv.org/abs/2309.06180) for the [KV cache](https://huggingface.co/blog/not-lain/kv-caching).

```bash

```

# Models