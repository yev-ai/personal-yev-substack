Basic containerization and TensorRT.

# Docker Installation

We should **install Docker Engine inside WSL** so that `nvidia-container-toolkit` can interface directly with the WSL Linux kernel via `dxgkrnl`. This lets WSL  sub in for a native host OS by directly managing the GPU resources.

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Add user to group
sudo usermod -aG docker $USER && newgrp docker

# Confirm socket (should show CONTAINER ID in red)
docker ps | grep "CONTAINER ID"
```

# vLLM Installation

Later on, we'll 

```bash

```