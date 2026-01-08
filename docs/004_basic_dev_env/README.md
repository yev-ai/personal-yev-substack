# Embeddings ...apparently?

This setup is a dumbed down version one of my startup's local-dev codebase indexing pipeline components (aka ingestion). It sits on top of [Jina AI's](https://jina.ai/) capable [jina-code-embeddings-0.5b](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) ([Link to ArXiv](https://arxiv.org/pdf/2508.21290)), which is based on Qwen2.5-Coder-0.5B. I benchmarked it since I'm sharing out my local development setup on Substack and wanted to objectively make sure that I'm not making my readers set up garbage. 

**These results were not what I expected** and I'm *extremely* skeptical about them until multiple 3rd party reviews. We're also comparing an [augmented model](./docker/Dockerfile.optimizer) against bare models, which is not apples to apples. 

So far, a buddy of mine at Cal tech ran it within 0.01% margin but n=1 is unhelpful. His main feedback was that the `CodeFeedBack-MT` scores are either invalid or brilliant due to the optimizer ...depending on how you want to look at it. We'd still have a (70.86%) average if we set CodeFeedback-MT score to 0.4. Please help pull down and verify or refute these results. Methodology and steps to reproduce below.

For reference, COIR [GitHub](https://github.com/CoIR-team/coir) and [Leaderboard](https://archersama.github.io/coir/).

| Benchmark | Our Setup | SFR-Embed-Code-2B_R | CodeSage-large-v2 | SFR-Embed-Code-400M_R | CodeSage-large |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Parameter Size (Millions) | 494 | 2000 | 1300 | **400** | 1300 |
| Average Total NDCG@10 | **0.7576** | 0.6741 | 0.6418 | 0.6189 | 0.6104 |
| Apps | **0.8395** | 0.7499 | 0.5045 | 0.4857 | 0.3416 |
| CosQA | **0.4141** | 0.3631 | 0.3273 | 0.3405 | 0.2859 |
| CodeSearchNet (Avg) | 0.8510 | 0.7350 | 0.9426 | 0.7253 | **0.9058** |
| CodeSearchNet-CCR (Avg) | **0.8983** | 0.8577 | 0.7809 | 0.8015 | 0.8436 |
| SyntheticText2sql | **0.6738** | 0.5900 | 0.5978 | 0.5896 | 0.5790 |
| CodeTrans-Contest | **0.9175** | 0.8663 | 0.8527 | 0.7567 | 0.8010 |
| CodeTrans-DL | **0.3721** | 0.3317 | 0.3329 | 0.3485 | 0.3345 |
| StackOverFlow QA | 0.8867 | **0.9054** | 0.7941 | 0.8951 | 0.7946 |
| CodeFeedBack-ST | **0.8328** | 0.8115 | 0.7132 | 0.7887 | 0.6606 |
| CodeFeedBack-MT | **0.8901** | 0.5308 | 0.5716 | 0.4575 | 0.5572 |
| Subcategory Score Details | ------- | ------- | ------- | ------- | ------- |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-go | 0.8869 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-java | 0.8347 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-js | 0.7765 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-ruby | 0.8021 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-py | 0.9598 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-php | 0.8461 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-CCR- go | 0.9210 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-CCR- java | 0.9044 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-CCR- js | 0.9009 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-CCR- ruby | 0.9090 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-CCR- py | 0.9090 | - | - | - | - |
| &nbsp;&nbsp;&nbsp;&nbsp;CodeSearchNet-CCR- php | 0.8457 | - | - | - | - |

# Oi... this turned into a pain.

If you're not coming from Substack, [this](https://www.yevelations.com/i/183747151/wsl-setup) and [this](https://www.yevelations.com/i/183748065/downloading-a-b-model) is what you want.

# Starting Up

```bash
# Download embedding model
hf download jinaai/jina-code-embeddings-0.5b

# Build TEI. This takes about 12-16 minutes if you're on the hardware in Chapter 1.
# I had to make one and build it cuz HF doesn't have a Blackwell-compatible release.
docker build -f docker/Dockerfile.embedding -t embedding-inference .

# Build proxy, throttler, and magic performance optimizer (also an abomination).
docker build -f docker/Dockerfile.optimizer -t embedding-optimizer .

# Start embedding model in separate Terminal window
chmod +x "$(git root)/docs/004_basic_dev_env/run-embedding-model.sh"
"$(git root)/docs/004_basic_dev_env/run-embedding-model.sh"

# Start embedding optimizer in separate Terminal window
chmod +x "$(git root)/docs/004_basic_dev_env/run-embedding-optimizer.sh"
"$(git root)/docs/004_basic_dev_env/run-embedding-optimizer.sh"
```

# Methodology

* Please thoroughly check this validity of [how I ran these benchmarks in this file](./coir-benchmark.py).
  * Related: [runner](./run-coir-benchmark.sh) and [container](./docker/Dockerfile.benchmark) (this repo's a mess, I'm just dumping it).
* You can also use it to replicate the above results. Details are in `/benchmarks`.

TODOs include not having multiple fatal edge cases and a trainwrecked mix of shell, python, and Dockerfiles.