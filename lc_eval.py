import time

from eval import load_questions, is_hit, PAUSE
from lc_search import search

def run(rows, rerank, store="numpy"): 
    hits = 0 
    for question, expected in rows: 
        docs = search(question, rerank=rerank, store=store)
        refs = [d.metadata["ref"] for d in docs]
        hit = any(is_hit(r, expected) for r in refs)
        hits += hit
        print(f"{'HIT' if hit else 'MISS'} {question[:45]:<45} {',' .join(refs)}")
        time.sleep(PAUSE[rerank])
    print(f"Hit rate: {hits}/{len(rows)}\n")
    return hits

if __name__ == "__main__": 
    rows = load_questions()
    print("=== Chroma: vector only ===")
    run(rows, rerank=False, store="chroma")
    print("=== Chroma: vector + reranker ===")
    run(rows, rerank=True, store="chroma")      