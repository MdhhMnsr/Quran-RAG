import sys
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain_core.runnables import RunnableLambda 

from load_verses import load_verses, text_to_embed, arabic_plain 

import chromadb
from langchain_chroma import Chroma

HERE = Path(__file__).parent

verses= load_verses()
vectors = np.load(HERE / "vectors.npy")
vectors_ar = np.load(HERE / "vectors_ar.npy")
embeddings = VoyageAIEmbeddings(model="voyage-4")
reranker = VoyageAIRerank(model="rerank-2.5", top_k=5)

chroma_store = Chroma(
    client=chromadb.PersistentClient(path=str(HERE / "chroma_db")),
    collection_name="quran_verses",
    embedding_function=embeddings,
)

class VerseRetriever(BaseRetriever):
    k: int = 30

    def _get_relevant_documents(self, query, *, run_manager):
        query_vector = np.array(embeddings.embed_query(query), dtype=np.float32)
        scores = np.maximum(vectors @ query_vector, vectors_ar @ query_vector)
        best = np.argsort(scores)[::-1] [:self.k]
        return [
            Document(
                page_content=text_to_embed(verses[i]) + "\n" + arabic_plain(verses[i]["arabic"]),
                metadata={**verses[i], "score": float(scores[i])},
            )
            for i in best
        ]

def search(question, k=5, candidates=30, rerank=True, store="numpy"):
    if store == "chroma":
        retriever = chroma_store.as_retriever(search_kwargs={"k":candidates})
    else:
        retriever = VerseRetriever(k=candidates)
    retriever = retriever.with_retry(
        stop_after_attempt=4,
        exponential_jitter_params={"initial":20},
    )

    docs = retriever.invoke(question)
    if not rerank:
        return docs[:k]
    rerank_step = RunnableLambda(
        lambda docs: reranker.compress_documents(docs,question)
    ).with_retry(stop_after_attempt=4, exponential_jitter_params={"initial": 20})
    return rerank_step.invoke(docs)

if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    docs = search(question)
    for doc in docs: 
        m = doc.metadata
        print(f"{m['ref']} (vector {m['score']:.3f}, rerank {m['relevance_score']:.3f}) {m['surah']}")
        print(f" {m['english']}\n")
    print(f"Rerank tokens: {docs[0].metadata['total_tokens']}")