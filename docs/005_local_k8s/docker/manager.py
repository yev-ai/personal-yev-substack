import uvicorn
import os
import sys
import logging
import httpx
import torch
import time
from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from transformers import AutoModel, AutoTokenizer
from torchao.quantization import quantize_, int4_weight_only

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SmartProxy")

# --- CONFIGURATION ---
def require_env(name: str, default: str = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        logger.error(f"❌ Required ENV var '{name}' not set")
        sys.exit(1)
    return value

TEI_BASE_URL = require_env("TEI_BASE_URL")
REAL_QDRANT_URL = "http://db-vector:6333"
PORT = int(os.environ.get("PORT", 8000))
MODEL_PATH = os.environ.get("MODEL_PATH", "jinaai/jina-reranker-v3")
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "4"))

QUERY_PREFIX = "Find the code snippet most similar to the query of:\n"
PASSAGE_PREFIX = "Candidate code snippet:\n"

LATEST_QUERY_TEXT = None
http_client = None
model = None
tokenizer = None

# --- LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, model, tokenizer
    
    # 1. Load Reranker (Native BF16 + TorchAO Quantization)
    logger.info(f"🚀 Loading Reranker ({MODEL_PATH}) with TorchAO...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # A. Load in full BF16 first (Blackwell Native)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )

        logger.info("🔨 Applying TorchAO Int4 Quantization...")
        quantize_(model, int4_weight_only())
        
        model.eval()
        logger.info("✅ Reranker Loaded & Quantized")
    except Exception as e:
        logger.error(f"❌ Failed to load reranker: {e}")

    # 2. Setup HTTP Client
    http_client = httpx.AsyncClient(
        timeout=120.0, 
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

# --- HELPER ---
def run_rerank_sync(query, candidates):
    return model.rerank(query, candidates)

# --- ROUTES ---

@app.get("/v1/models")
async def list_models():
    # logger.info("🔍 Handshake: Roo requested /v1/models")
    return {
        "object": "list",
        "data": [
            {"id": "jina-code-embeddings", "object": "model", "created": 1686935002, "owned_by": "jina-manager"},
            {"id": "jina-reranker-v3", "object": "model", "created": 1686935002, "owned_by": "jina-manager"}
        ]
    }

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    raw_input = body.get("input")
    if not raw_input: raise HTTPException(status_code=400, detail="Missing input")
    
    global LATEST_QUERY_TEXT
    texts = [raw_input] if isinstance(raw_input, str) else raw_input
    
    # Save query logic
    if len(texts) == 1 and len(texts[0]) < 2000:
        LATEST_QUERY_TEXT = texts[0]

    processed_inputs = []
    for t in texts:
        if len(texts) == 1 and len(t) < 2000:
             processed_inputs.append(QUERY_PREFIX + t)
        else:
             processed_inputs.append(PASSAGE_PREFIX + t)

    try:
        resp = await http_client.post(f"{TEI_BASE_URL}/embed", json={"inputs": processed_inputs, "truncate": True})
        resp.raise_for_status()
        embeddings = resp.json()
    except Exception as e:
        logger.error(f"❌ TEI Embedder Failed: {e}")
        raise HTTPException(status_code=500, detail="Embedder Failed")

    total_chars = sum(len(t) for t in texts)
    approx_tokens = max(1, total_chars // 4)
    
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(embeddings)],
        "model": "jina-code-embeddings",
        "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens}
    }

@app.post("/collections/{collection_name}/points/search")
async def proxy_qdrant_search(collection_name: str, request: Request):
    global LATEST_QUERY_TEXT
    logger.info(f"🔎 Search: {collection_name}")
    
    try:
        body = await request.json()
        original_limit = body.get("limit", 20)
        body["limit"] = 100 
        body["with_payload"] = True 
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        q_res = await http_client.post(f"{REAL_QDRANT_URL}/collections/{collection_name}/points/search", json=body)
        q_res.raise_for_status()
        data = q_res.json()
        results = data.get("result", [])
    except Exception as e:
        logger.error(f"❌ Qdrant Search Failed: {e}")
        raise HTTPException(status_code=502, detail="Qdrant Failed")

    if LATEST_QUERY_TEXT and results and model:
        candidates = []
        valid_indices = []
        for i, hit in enumerate(results):
            payload = hit.get("payload", {})
            text = payload.get("text") or payload.get("content") or payload.get("snippet") or payload.get("code")
            if text:
                candidates.append(text)
                valid_indices.append(i)
        
        if candidates:
            logger.info(f"✨ Reranking {len(candidates)} candidates")
            try:
                all_scores = []
                for i in range(0, len(candidates), RERANK_BATCH_SIZE):
                    batch = candidates[i : i + RERANK_BATCH_SIZE]
                    batch_scores = await run_in_threadpool(run_rerank_sync, LATEST_QUERY_TEXT, batch)
                    all_scores.extend(batch_scores)

                reranked_hits = []
                for item in all_scores:
                    idx = item["index"]
                    original_idx = valid_indices[idx]
                    hit = results[original_idx]
                    hit["score"] = float(item["relevance_score"])
                    reranked_hits.append(hit)
                
                reranked_hits.sort(key=lambda x: x["score"], reverse=True)
                data["result"] = reranked_hits[:original_limit]
                return data

            except Exception as e:
                logger.error(f"⚠️ Reranking Failed: {e}")
                data["result"] = results[:original_limit]
                return data

    data["result"] = results[:original_limit]
    return data
# --- ROBUST CATCH-ALL PROXY ---
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"])
async def catch_all_proxy(request: Request, path_name: str):
    req_id = str(int(time.time() * 1000))[-6:]
    target_url = f"{REAL_QDRANT_URL}/{path_name}"
    
    logger.info(f"[{req_id}] 📥 {request.method} /{path_name}")

    # Headers Cleanup - exclude accept-encoding to prevent gzip issues
    excluded = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}
    clean_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}
    clean_headers["Accept-Encoding"] = "identity"  # Force no compression from Qdrant

    # Input Handling
    content = None
    if request.method in ["POST", "PUT", "PATCH"]:
        async def body_stream():
            async for chunk in request.stream():
                yield chunk
        content = body_stream()

    try:
        resp = await http_client.request(
            method=request.method,
            url=target_url,
            content=content,
            params=request.query_params,
            headers=clean_headers
        )
        
        logger.info(f"[{req_id}] ⬅️ Status: {resp.status_code}")

        # Workaround: Treat 409 on collection PUT as success (idempotent create)
        if resp.status_code == 409 and request.method == "PUT" and path_name.startswith("collections/"):
            logger.info(f"[{req_id}] ⚠️ Converting 409 Conflict to 200 OK (collection already exists)")
            return Response(
                content=b'{"result":true,"status":"ok"}',
                status_code=200,
                headers={"Content-Type": "application/json", "Connection": "close"}
            )

        # Prepare Response Headers - also exclude content-encoding to prevent decompression errors
        resp_headers = {k: v for k, v in resp.headers.items() 
                        if k.lower() not in {"content-length", "transfer-encoding", "connection", "content-encoding"}}
        
        resp_headers["Connection"] = "close"

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers
        )

    except Exception as e:
        logger.error(f"[{req_id}] ❌ PROXY FAIL: {e}")
        raise HTTPException(status_code=502, detail=f"Proxy Failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")