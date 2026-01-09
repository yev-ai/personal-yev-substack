import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

# --- CONFIGURATION ---
MODEL_ID = "jinaai/jina-reranker-v3"
PORT = 8081

app = FastAPI(title="Jina V3 Optimized Reranker")

# --- LOAD MODEL (Blazing Fast Mode) ---
print(f"🚀 Loading {MODEL_ID} with Flash Attention 2...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,        # Half-precision (Faster)
        trust_remote_code=True,
        attn_implementation="flash_attention_2", # The "Speed" Secret
        device_map="cuda"
    )
    model.eval()
    
    # JIT Compile (Optional: Adds ~60s startup time but speeds up queries)
    # Comment this out if startup time is annoying, but keep it for max speed.
    print("🔥 Compiling model with torch.compile (this takes a moment)...")
    model = torch.compile(model, mode="max-autotune")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# --- DATA MODELS ---
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

# --- ENDPOINTS ---
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if not req.documents:
        return {"results": []}

    # Prepare pairs for the model [ [query, doc1], [query, doc2] ... ]
    pairs = [[req.query, doc] for doc in req.documents]
    
    start_time = time.time()
    
    try:
        with torch.inference_mode():
            # Tokenize
            inputs = tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=1024  # Cap context to prevent OOM
            ).to("cuda")

            # Inference
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1).float().cpu().numpy()
            
        # Format Results
        results = []
        for i, score in enumerate(scores):
            results.append({
                "index": i,
                "relevance_score": float(score),
                "document": {"text": req.documents[i]}
            })

        # Sort and Slice
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = results[:req.top_n]
        
        # Log speed for sanity check
        print(f"⚡ Reranked {len(req.documents)} docs in {time.time() - start_time:.3f}s")
        
        return {"results": results}

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)