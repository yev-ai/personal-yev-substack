import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
from contextlib import asynccontextmanager
import httpx
import logging
import time
import sys
import os

# --- CONFIGURATION ---
def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Required ENV var '{name}' not set", file=sys.stderr)
        sys.exit(1)
    return value

MODEL_ID = os.environ.get("MODEL_ID", "jinaai/jina-reranker-v3")
TEI_BASE_URL = require_env("TEI_BASE_URL")
PORT = int(require_env("PORT"))

# Optional: Use local model path if mounted
MODEL_PATH = os.environ.get("MODEL_PATH", MODEL_ID)

# Default: "Find the most relevant code snippet given the following query:\n"
QUERY_PREFIX = "Find the code snippet most similar to the query of:\n"
PASSAGE_PREFIX = "Candidate code snippet:\n"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("JinaManager")
http_client = None

# --- LOAD LOCAL RERANKER MODEL ---
print(f"🚀 Loading {MODEL_PATH} with Flash Attention 2...")
try:
    # 1. Load Tokenizer & CRITICAL FIX: Set Padding Token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # <--- Fixes crash on batching

    # 2. Load Model (AutoModel loads the custom JinaForRanking class)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )
    model.eval()

except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
    http_client = httpx.AsyncClient(base_url=TEI_BASE_URL, timeout=60.0, limits=limits)
    print(f"Jina manager connected to TEI embedder at {TEI_BASE_URL}")
    print(f"Manager ready on port {PORT}")
    yield
    await http_client.aclose()

app = FastAPI(title="Jina Manager - Local Reranker + Proxy Embedder", lifespan=lifespan)

# --- DATA MODELS ---
class EmbeddingRequest(BaseModel):
    input: list[str] | str
    model: str | None = "default"

class RerankRequest(BaseModel):
    model: str = "jina-reranker-v3"
    query: str
    documents: List[str]
    top_n: int = 10

class DocumentObject(BaseModel):
    text: str

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: DocumentObject

class RerankResponse(BaseModel):
    results: List[RerankResult]

# --- HELPER FUNCTIONS ---
def is_query(text: str) -> bool:
    if len(text) > 300:
        return False
    if text.count("\n") > 2:
        return False
    text_lower = text.lower()
    if "?" in text or "how " in text_lower or "find " in text_lower or "what " in text_lower:
        return True
    return True

# --- ENDPOINTS ---

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    processed_inputs = [
        (QUERY_PREFIX + t) if is_query(t) else (PASSAGE_PREFIX + t)
        for t in texts
    ]
    try:
        response = await http_client.post("/embed", json={"inputs": processed_inputs, "truncate": True})
        response.raise_for_status()
        embeddings = response.json()
    except Exception as e:
        logger.error(f"TEI Connection Failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding Engine Unavailable")

    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(embeddings)],
        "model": "jina-code-embeddings"
    }

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if not req.documents:
        return {"results": []}

    start_time = time.time()

    try:
        # Use the native rerank method (handles sorting, batching, and unique attention patterns)
        # It returns a list of dicts: {'index': int, 'relevance_score': float, 'document': str}
        results = model.rerank(req.query, req.documents, top_n=req.top_n)

        # Map to your response format
        formatted_results = []
        for res in results:
            formatted_results.append({
                "index": res['index'],
                "relevance_score": float(res['relevance_score']),
                "document": {"text": req.documents[res['index']]} 
            })

        print(f"⚡ Reranked {len(req.documents)} docs in {time.time() - start_time:.3f}s")
        return {"results": formatted_results}

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")