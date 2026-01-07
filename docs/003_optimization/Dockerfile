FROM nvidia/cuda:12.9.0-devel-ubuntu24.04
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

# (Very ghetto) patch to fix AttributeError in vLLM V1 FlashInfer backend for Speculative Decoding (a bug in nightly)
RUN sed -i "s/num_kv_heads=kv_cache_spec.num_kv_heads,/num_kv_heads=getattr(kv_cache_spec, 'num_kv_heads', 0),/g" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]