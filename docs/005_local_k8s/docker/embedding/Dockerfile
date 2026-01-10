FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS base-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libssl-dev \
    pkg-config \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

ENV SCCACHE="0.10.0" \
    SCCACHE_DIR="/root/.cache/sccache" \
    PATH="/root/.cargo/bin:${PATH}"

RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

ENV CARGO_CHEF="0.1.71"
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    cargo install cargo-chef --version $CARGO_CHEF --locked

WORKDIR /usr/src
ARG TEI_COMMIT="0ec2ba5073de67c016cf77b8942ddb15ed3b0397"
RUN git clone https://github.com/huggingface/text-embeddings-inference . && \
    git checkout $TEI_COMMIT
COPY ./sm120.patch .
RUN patch -p1 < sm120.patch

FROM base-builder AS builder
WORKDIR /usr/src
ARG CUDA_COMPUTE_CAP="120"
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} \
    RUSTC_WRAPPER="/usr/local/bin/sccache" \
    RUSTFLAGS="-C target-cpu=native" \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1" \
    CARGO_PROFILE_RELEASE_LTO="fat" \
    TORCH_CUDA_ARCH_LIST="12.0" \
    CUDA_ARCH_LIST="12.0" \
    CUDA_HOME="/usr/local/cuda"

RUN nvprune --generate-code code=sm_$CUDA_COMPUTE_CAP /usr/local/cuda/lib64/libcublas_static.a -o /usr/local/cuda/lib64/libcublas_static.a

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo generate-lockfile

ARG CUTLASS_COMMIT="4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b"

RUN --mount=type=cache,target=/root/.cargo/git \
    find /root/.cargo/git -name "build.rs" -path "*flash-attn*" \
    -exec sed -i "s/7d49e6c7e2f8896c47f586706e67e1fb215529dc/${CUTLASS_COMMIT}/g" {} \; && \
    echo "Patched CUTLASS commit to ${CUTLASS_COMMIT}"

RUN --mount=type=cache,target=/root/.cache/sccache \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/usr/src/target \
    cargo build --release --bin text-embeddings-router \
    -F candle-cuda -F flash-attn -F http --no-default-features && \
    sccache -s && \
    cp /usr/src/target/release/text-embeddings-router /usr/local/bin/ && \
    strip --strip-all /usr/local/bin/text-embeddings-router

FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04 AS base

ENV HUGGINGFACE_HUB_CACHE="/data" \
    USE_FLASH_ATTENTION="True" \
    PORT=8000

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]
