import uvicorn
import os
import sys
import logging
import httpx
import torch
import time
import asyncio
import hashlib
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from transformers import AutoModel
from torchao.quantization import quantize_, Int4WeightOnlyConfig

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SmartProxy")

def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error(f"Required ENV var '{name}' not set")
        sys.exit(1)
    return value

TEI_BASE_URL = require_env("TEI_BASE_URL")
VECTOR_DB_BASE_URL = require_env("VECTOR_DB_BASE_URL")
PORT = int(os.environ.get("PORT", 8000))
MODEL_PATH = os.environ.get("MODEL_PATH", "jinaai/jina-reranker-v3")
RERANK_BATCH_SIZE = 64 # Use listwise arch now that we implemented it

QUERY_PREFIX = "Find the code snippet most similar to the query of:\n"
PASSAGE_PREFIX = "Candidate code snippet:\n"


class QueryCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        self.cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    def _hash_vector(self, vector: list) -> str:
        key = ",".join(f"{v:.6f}" for v in vector[:32])
        # Like unc Fowler says, it's jus cache invalidation and namin thangz
        return hashlib.md5(key.encode()).hexdigest()
    
    def _evict_expired(self) -> int:
        now = time.time()
        expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
        for k in expired:
            del self.cache[k]
        return len(expired)
    
    async def store(self, vector: list, query_text: str) -> None:
        async with self._lock:
            evicted = self._evict_expired()
            if evicted > 0:
                logger.debug(f"Evicted {evicted} expired cache entries")
            
            h = self._hash_vector(vector)
            self.cache[h] = (query_text, time.time())
            self.cache.move_to_end(h)
            
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
                logger.debug(f"LRU eviction triggered, cache size: {self.max_size}")
            
            preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
            logger.debug(f"Cached query: {preview}")
    
    async def get(self, vector: list) -> str | None:
        async with self._lock:
            self._evict_expired()
            h = self._hash_vector(vector)
            entry = self.cache.get(h)
            
            if entry:
                self._hits += 1
                self.cache.move_to_end(h)
                logger.debug(f"Cache hit (hits: {self._hits}, misses: {self._misses})")
                return entry[0]
            
            self._misses += 1
            logger.debug(f"Cache miss (hits: {self._hits}, misses: {self._misses})")
            return None
    
    async def stats(self) -> dict:
        async with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
            }


query_cache = QueryCache(max_size=1000, ttl_seconds=60)
http_client = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, model
    
    logger.info(f"Loading Reranker ({MODEL_PATH}) with TorchAO...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )

        logger.info("Applying TorchAO Int4 Quantization...")
        quantize_(model, Int4WeightOnlyConfig(group_size=128))
        
        model.eval()
        logger.info("Reranker Loaded & Quantized")
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")

    http_client = httpx.AsyncClient(
        timeout=120.0, 
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

def run_rerank_sync(query, candidates):
    return model.rerank(query, candidates)

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "jina-code-embeddings", "object": "model", "created": 1686935002, "owned_by": "jina-manager"},
            {"id": "jina-reranker-v3", "object": "model", "created": 1686935002, "owned_by": "jina-manager"}
        ]
    }

@app.get("/v1/cache/stats")
async def cache_stats():
    return await query_cache.stats()

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    raw_input = body.get("input")
    if not raw_input:
        raise HTTPException(status_code=400, detail="Missing input")
    
    texts = [raw_input] if isinstance(raw_input, str) else raw_input
    
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input array")
    
    is_query = len(texts) == 1 and len(texts[0]) < 2000
    original_query_text = texts[0] if is_query else None

    processed_inputs = []
    for t in texts:
        if is_query:
            processed_inputs.append(QUERY_PREFIX + t)
        else:
            processed_inputs.append(PASSAGE_PREFIX + t)

    all_embeddings = []
    total_items = len(processed_inputs)
    
    if total_items > RERANK_BATCH_SIZE:
        logger.info(f"Chunking {total_items} inputs into batches of {RERANK_BATCH_SIZE}")
    
    try:
        for batch_start in range(0, total_items, RERANK_BATCH_SIZE):
            batch_end = min(batch_start + RERANK_BATCH_SIZE, total_items)
            batch_inputs = processed_inputs[batch_start:batch_end]
            
            resp = await http_client.post(
                f"{TEI_BASE_URL}/embed",
                json={"inputs": batch_inputs, "truncate": True}
            )
            resp.raise_for_status()
            batch_embeddings = resp.json()
            all_embeddings.extend(batch_embeddings)
            
            if total_items > RERANK_BATCH_SIZE:
                logger.debug(f"Processed batch {batch_start // RERANK_BATCH_SIZE + 1}/{(total_items + RERANK_BATCH_SIZE - 1) // RERANK_BATCH_SIZE}")
    except httpx.HTTPStatusError as e:
        logger.error(f"TEI Embedder Failed: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Embedder Failed: {e.response.status_code}")
    except Exception as e:
        logger.error(f"TEI Embedder Failed: {e}")
        raise HTTPException(status_code=500, detail="Embedder Failed")

    if is_query and original_query_text and all_embeddings and len(all_embeddings) > 0:
        await query_cache.store(all_embeddings[0], original_query_text)

    total_chars = sum(len(t) for t in texts)
    approx_tokens = max(1, total_chars // 4)
    
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(all_embeddings)],
        "model": "jina-code-embeddings",
        "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens}
    }

@app.post("/collections/{collection_name}/points/search")
async def proxy_qdrant_search(collection_name: str, request: Request):
    logger.info(f"Search: {collection_name}")
    
    try:
        body = await request.json()
        original_limit = body.get("limit", 20)
        original_score_threshold = body.pop("score_threshold", None)
        logger.info(f"Request received. Limit {original_limit}, Score Threshold: {original_score_threshold}")
        body["limit"] = 100 
        body["with_payload"] = True 
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    search_vector = body.get("vector")
    query_text = None
    
    if search_vector:
        if isinstance(search_vector, list) and len(search_vector) > 0 and isinstance(search_vector[0], (int, float)):
            query_text = await query_cache.get(search_vector)
        elif isinstance(search_vector, dict) and "vector" in search_vector:
            query_text = await query_cache.get(search_vector["vector"])
    
    if query_text:
        logger.info(f"Correlated query: {query_text[:50]}...")
    else:
        logger.warning("No query text found in cache - reranking will be skipped")

    try:
        q_res = await http_client.post(f"{VECTOR_DB_BASE_URL}/collections/{collection_name}/points/search", json=body)
        q_res.raise_for_status()
        data = q_res.json()
        results = data.get("result", [])
    except Exception as e:
        logger.error(f"Qdrant Search Failed: {e}")
        raise HTTPException(status_code=502, detail="Qdrant Failed")

    if query_text and results and model:
        candidates = []
        valid_indices = []
        for i, hit in enumerate(results):
            payload = hit.get("payload", {})
            text = payload.get("text") or payload.get("content") or payload.get("snippet") or payload.get("code")
            
            file_path = payload.get("file_path") or payload.get("path") or payload.get("filename")
            if text and file_path:
                text = f"File: {file_path}\n{text}"
            
            if text:
                candidates.append(text)
                valid_indices.append(i)
        
        if candidates:
            logger.info(f"Reranking {len(candidates)} candidates")
            try:
                all_scores = []
                for batch_start in range(0, len(candidates), RERANK_BATCH_SIZE):
                    batch = candidates[batch_start : batch_start + RERANK_BATCH_SIZE]
                    batch_scores = await run_in_threadpool(run_rerank_sync, query_text, batch)
                    for item in batch_scores:
                        item["index"] = item["index"] + batch_start
                    all_scores.extend(batch_scores)

                reranked_hits = []
                for item in all_scores:
                    idx = item["index"]
                    original_idx = valid_indices[idx]
                    hit = results[original_idx]
                    hit["score"] = float(item["relevance_score"])
                    reranked_hits.append(hit)
                
                reranked_hits.sort(key=lambda x: x["score"], reverse=True)
                
                if original_score_threshold is not None:
                    reranked_hits = [hit for hit in reranked_hits if hit["score"] >= original_score_threshold]
                    logger.info(f"After threshold filter ({original_score_threshold}): {len(reranked_hits)} results")
                
                data["result"] = reranked_hits[:original_limit]
                return data

            except Exception as e:
                logger.error(f"Reranking Failed: {e}")
                if original_score_threshold is not None:
                    results = [hit for hit in results if hit.get("score", 0) >= original_score_threshold]
                data["result"] = results[:original_limit]
                return data

    if original_score_threshold is not None:
        results = [hit for hit in results if hit.get("score", 0) >= original_score_threshold]
        logger.info(f"After threshold filter ({original_score_threshold}): {len(results)} results")
    
    data["result"] = results[:original_limit]
    return data

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"])
async def catch_all_proxy(request: Request, path_name: str):
    req_id = str(int(time.time() * 1000))[-6:]
    target_url = f"{VECTOR_DB_BASE_URL}/{path_name}"
    
    logger.info(f"[{req_id}] {request.method} /{path_name}")

    excluded = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}
    clean_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}
    clean_headers["Accept-Encoding"] = "identity"

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
        
        logger.info(f"[{req_id}] -> Status: {resp.status_code}")

        if resp.status_code == 409 and request.method == "PUT" and path_name.startswith("collections/"):
            logger.info(f"[{req_id}] Converting 409 Conflict to 200 OK (collection already exists)")
            return Response(
                content=b'{"result":true,"status":"ok"}',
                status_code=200,
                headers={"Content-Type": "application/json", "Connection": "close"}
            )

        resp_headers = {k: v for k, v in resp.headers.items() 
                        if k.lower() not in {"content-length", "transfer-encoding", "connection", "content-encoding"}}
        
        resp_headers["Connection"] = "close"

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers
        )

    except Exception as e:
        logger.error(f"[{req_id}] V2 PROXY FAIL: {e}")
        raise HTTPException(status_code=502, detail=f"Proxy Failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
