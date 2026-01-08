import logging
import asyncio
import httpx
import numpy as np
from coir.evaluation import COIR
from coir.data_loader import get_tasks

PROXY_URL = "http://127.0.0.1:1335/v1/embeddings"
MODEL_ID = "jina-code-embeddings"

class ProxyModel:
    def __init__(self, model_name="local-proxy"):
        self.model_name = model_name
        self.limits = httpx.Limits(max_keepalive_connections=50, max_connections=1000)
        self.timeout = httpx.Timeout(360.0, connect=10.0) 

    async def _send_single_async(self, client, text, index, semaphore, delay=0.0):
        if delay > 0:
            await asyncio.sleep(delay)

        async with semaphore: 
            retries = 5
            for attempt in range(retries):
                try:
                    response = await client.post(
                        PROXY_URL,
                        json={"input": text, "model": MODEL_ID},
                    )
                    response.raise_for_status()
                    data = response.json()["data"][0]["embedding"]
                    return index, data
                
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [500, 502, 503, 504]:
                        wait_time = 1.0 * (2 ** attempt)
                        if attempt > 1: 
                            print(f"\n500 Error on query {index}. Retry {attempt}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"\nFatal Error on query {index}: {e}")
                        return index, np.zeros(768)
                except Exception as e:
                    return index, np.zeros(768)
            
            return index, np.zeros(768)

    def encode_queries(self, queries, batch_size=100, **kwargs):
        print(f"Encoding {len(queries)} queries (Ramping up to {batch_size} concurrent)...")
        
        async def run_all():
            sem = asyncio.Semaphore(batch_size)
            
            async with httpx.AsyncClient(limits=self.limits, timeout=self.timeout) as client:
                tasks = []
                
                ramp_duration = 5.0 
                
                for i, q in enumerate(queries):
                    delay = 0.0
                    if i < batch_size * 2:
                        delay = (i / (batch_size * 2)) * ramp_duration
                    
                    tasks.append(
                        self._send_single_async(client, q, i, sem, delay=delay)
                    )
                
                results = []
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    res = await coro
                    results.append(res)
                    if i % 50 == 0 or i == len(queries) - 1:
                        print(f"   Query {i+1}/{len(queries)}...", end="\r")

                return results

        raw_results = asyncio.run(run_all())
        raw_results.sort(key=lambda x: x[0])
        return np.array([x[1] for x in raw_results])

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
            if isinstance(corpus[0], dict):
                texts = [doc.get("text", "") + " " + doc.get("title", "") for doc in corpus]
            else:
                texts = corpus

            embeddings = []
            print(f"📚 Encoding {len(texts)} docs (Batch Size: {batch_size})...")

            with httpx.Client(timeout=self.timeout, limits=self.limits) as client:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    retries = 3
                    success = False
                    
                    for attempt in range(retries):
                        try:
                            response = client.post(
                                PROXY_URL,
                                json={"input": list(batch), "model": MODEL_ID}
                            )
                            response.raise_for_status()
                            data = response.json()["data"]
                            embeddings.extend([item["embedding"] for item in data])
                            success = True
                            break
                        
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code in [500, 502, 503, 504]:
                                wait_time = 2.0 * (2 ** attempt)
                                print(f"\nBatch {i} hit Server Error. Retrying in {wait_time}s...", end="")
                                import time; time.sleep(wait_time)
                            else:
                                print(f"\nFatal Error in batch {i}: {e}")
                                break
                        except Exception as e:
                            print(f"\nNetwork glitch in batch {i}: {e}. Retrying...")
                            import time; time.sleep(1)

                    if not success:
                        print(f"\nSKIPPING BATCH {i} (All retries failed).")
                        embeddings.extend([np.zeros(768).tolist() for _ in batch])
                    
                    print(f"   Doc {len(embeddings)}/{len(texts)}...", end="\r")

            print("\nFinished encoding corpus.")
            return np.array(embeddings)

if __name__ == "__main__":
    print("Starting CoIR Eval via Proxy (Final Robust Config)...")
    model = ProxyModel()
    model.encode_queries = lambda q, batch_size=None, **k: ProxyModel.encode_queries(model, q, batch_size=256, **k)
    model.encode_corpus = lambda c, batch_size=None, **k: ProxyModel.encode_corpus(model, c, batch_size=256, **k)
    # [
    #     "codefeedback-mt",
    #     "codesearchnet", apps, cosqa
    #     "synthetic-text2sql",
    #     "codesearchnet-ccr"
    #     "codetrans-contest",
    #     "codetrans-dl",
    #     "stackoverflow-qa",
    #     "codefeedback-st",
    # ]
    all_tasks = get_tasks(tasks=["codefeedback-mt"])

    # target_lang = "javascript" <-- Optional filter. Ran it on all languages.
    # tasks = {k: v for k, v in all_tasks.items() if target_lang in k.lower()}
    # if not tasks:
    #         print(f"❌ Could not find task for language: {target_lang}")
    #         exit()

    # print(f"🎯 Selected Task: {list(tasks.keys())}")
    evaluation = COIR(tasks=all_tasks, batch_size=768) # Reduce to 48 and 16/16 above *ONLY* for stackoverflow-qa
    results = evaluation.run(model, output_folder="results/local-jina-0.5b")
    print("\n🏆 FINAL RESULTS:")
    print(results)