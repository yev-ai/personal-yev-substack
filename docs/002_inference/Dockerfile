FROM nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG VLLM_WHEEL

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ${VLLM_WHEEL} /app/

RUN python3 -m pip install /app/${VLLM_WHEEL}
RUN python3 -m pip install flash-attn --no-build-isolation

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]