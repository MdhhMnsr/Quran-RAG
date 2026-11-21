import sys
from pathlib import Path

import numpy as np
import voyageai

from load_verses import load_verses, text_to_embed , PRICE_PER_MILLION 

HERE = Path(__file__).parent
MODEL = "voyage-4"
RERANK_MODEL ="rerank-2.5"

verses =load_verses()
vectors = np.load(HERE / "vectors.npy")
vo = voyageai.Client(max_retries=3)

def search(question, k=5, candidates = 30, rerank=True):
    result = vo.embed([question], model=MODEL, input_type="query")
    query = np.array(result.embeddings[0], dtype=np.float32)
    scores = vectors @ query
    best = np.argsort(scores)[::-1][:candidates]
    
    if not rerank: 
         found = [{**verses[i], "score": float(scores[i])} for i in best[:k]] 
         return found, result.total_tokens

    documents = [text_to_embed(verses[i]) for i in best]
    reranked = vo.rerank(question, documents, model=RERANK_MODEL, top_k=k)

    found = []
    for r in reranked.results:
         i = best[r.index]
         found.append({**verses[i], "score": float(scores[i]), "rerank": r.relevance_score})
    return found, result.total_tokens + reranked.total_tokens  
 
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) 
    found, tokens = search(question)
    for verse in found: 
            print(f"{verse['ref']} (vector {verse['score']:.3f}, rerank {verse['rerank']:.3f}) {verse['surah']}")
            print(f"  {verse['english']}\n")
    print(f"Tokens: {tokens}")
    print(f"Value at list price: ${tokens / 1_000_000 * PRICE_PER_MILLION:.6f}")
