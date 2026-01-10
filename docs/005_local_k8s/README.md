In this directory:


```bash

docker build --progress=plain -f docker/Dockerfile.manager -t embedding-manager .
docker build --progress=plain -f docker/Dockerfile.embedding -t embedding-inference .


git clone https://github.com/Helicone/helicone.git
```

In ./helicone/docker/dockerfiles/dockerfile_web, on Line 10, add:
`COPY packages/pricing/package.json ./packages/pricing/`

So it looks like:

```
FROM node:20-bookworm-slim AS builder

WORKDIR /app

COPY package.json yarn.lock ./
COPY packages/common/package.json ./packages/common/
COPY packages/cost/package.json ./packages/cost/
COPY packages/filters/package.json ./packages/filters/
COPY packages/llm-mapper/package.json ./packages/llm-mapper/
COPY packages/prompts/package.json ./packages/prompts/
COPY packages/secrets/package.json ./packages/secrets/
COPY packages/pricing/package.json ./packages/pricing/
COPY web/package.json ./web/
COPY valhalla/jawn/package.json ./valhalla/jawn/
COPY bifrost/package.json ./bifrost/
```