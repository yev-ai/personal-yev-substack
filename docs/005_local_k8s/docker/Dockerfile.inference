FROM nvcr.io/nvidia/pytorch:25.12-py3 AS header_donor
FROM nvcr.io/nvidia/vllm:25.12.post1-py3 AS builder

ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_ARCH_LIST="12.0"
ENV VLLM_TARGET_DEVICE="cuda"

ENV VLLM_USE_PRECOMPILED_FLASHINFER=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_BUILD_ISOLATION=1

WORKDIR /build

COPY --from=header_donor /usr/local/cuda/include /usr/local/cuda/include
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages setuptools_scm packaging wheel
RUN git clone https://github.com/vllm-project/vllm.git . && \
    git checkout 1963245ed15a9f232ac3b763c7a03351d77ae799

RUN cat <<'PATCH' | patch -p1
diff --git a/pyproject.toml b/pyproject.toml
index a5d41c673..c5d3d85ef 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -6,7 +6,7 @@ requires = [
     "packaging>=24.2",
     "setuptools>=77.0.3,<81.0.0",
     "setuptools-scm>=8.0",
-    "torch == 2.9.1",
+    "torch",
     "wheel",
     "jinja2",
     "grpcio-tools>=1.76.0",
diff --git a/requirements/build.txt b/requirements/build.txt
index b3ef0a710..69f1a70a6 100644
--- a/requirements/build.txt
+++ b/requirements/build.txt
@@ -4,7 +4,7 @@ ninja
 packaging>=24.2
 setuptools>=77.0.3,<81.0.0
 setuptools-scm>=8
-torch==2.9.1
+torch
 wheel
 jinja2>=3.1.6
 regex
diff --git a/requirements/cuda.txt b/requirements/cuda.txt
index 1417fb991..945ed2c89 100644
--- a/requirements/cuda.txt
+++ b/requirements/cuda.txt
@@ -1,13 +1,10 @@
 # Common dependencies
 -r common.txt
 
-numba == 0.61.2 # Required for N-gram speculative decoding
+numba
 
 # Dependencies for NVIDIA GPUs
 ray[cgraph]>=2.48.0 # Ray Compiled Graph, required for pipeline parallelism in V1.
-torch==2.9.1
-torchaudio==2.9.1
-# These must be updated alongside torch
-torchvision==0.24.1 # Required for phi3v processor. See https://github.com/pytorch/vision?tab=readme-ov-file#installation for corresponding version
-# FlashInfer should be updated together with the Dockerfile
-flashinfer-python==0.5.3
+torch
+torchaudio
+torchvision
PATCH


RUN export MAX_JOBS=28 && \
    export USE_NINJA=1 && \
    python3 setup.py bdist_wheel --dist-dir=dist

FROM nvcr.io/nvidia/vllm:25.12.post1-py3

WORKDIR /app

COPY --from=builder /build/dist/*.whl /app/

RUN apt-get update && apt-get install -y zip unzip && rm -rf /var/lib/apt/lists/*

RUN WHEEL_NAME=$(ls *.whl) && \
    mkdir -p edgelord-inline-fix/vllm-info && \
    unzip "$WHEEL_NAME" "vllm-*.dist-info/METADATA" -d edgelord-inline-fix && \
    sed -i '/Requires-Dist: torch/d' edgelord-inline-fix/vllm-*.dist-info/METADATA && \
    sed -i '/Requires-Dist: flashinfer/d' edgelord-inline-fix/vllm-*.dist-info/METADATA && \
    sed -i '/Requires-Dist: flash-attn/d' edgelord-inline-fix/vllm-*.dist-info/METADATA && \
    sed -i '/Requires-Dist: xformers/d' edgelord-inline-fix/vllm-*.dist-info/METADATA && \
    sed -i '/Requires-Dist: triton/d' edgelord-inline-fix/vllm-*.dist-info/METADATA && \
    cd edgelord-inline-fix && zip -u "../$WHEEL_NAME" vllm-*.dist-info/METADATA && cd .. && \
    rm -rf edgelord-inline-fix

RUN pip install --break-system-packages ijson httpx-sse sse-starlette grpcio-reflection mcp fastrlock pyjwt cryptography && \
    pip install --break-system-packages "xgrammar==0.1.29" "compressed-tensors==0.13.0" --no-deps

RUN pip uninstall -y vllm --break-system-packages && \
    pip install /app/*.whl --no-deps --no-cache-dir --break-system-packages && \
    rm /app/*.whl

RUN sed -i "s/num_kv_heads=kv_cache_spec.num_kv_heads,/num_kv_heads=getattr(kv_cache_spec, 'num_kv_heads', 0),/g" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py

RUN sed -i 's/inductor_config_patches\["assume_32bit_indexing"\] = True/pass/' /usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py && \
    rm -rf /usr/local/lib/python3.12/dist-packages/vllm/compilation/__pycache__

RUN python3 -c "import vllm;        print(f'vLLM:       {vllm.__version__}'); assert '0.14' in vllm.__version__, 'vLLM Version Mismatch!'"
RUN python3 -c "import torch;       print(f'Torch:      {torch.__version__}'); assert '2.10.0a0' in torch.__version__, 'Torch Downgraded!'"
RUN python3 -c "import torchvision; print(f'Vision:     {torchvision.__version__}'); assert '0.25.0a0' in torchvision.__version__, 'Vision Downgraded!'"
RUN python3 -c "import xformers;    print(f'xFormers:   {xformers.__version__}'); assert 'nv25.12' in xformers.__version__, 'xFormers Downgraded!'"
RUN python3 -c "import flashinfer;  print(f'FlashInfer: {flashinfer.__version__}'); assert '0.6.0rc2' in flashinfer.__version__, 'FlashInfer Downgraded!'"
RUN python3 -c "import flash_attn;  print(f'FlashAttn:  {flash_attn.__version__}'); assert '2.7.4' in flash_attn.__version__, 'FlashAttn Downgraded!'"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]