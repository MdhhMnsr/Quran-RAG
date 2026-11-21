from pathlib import Path

import chromadb
import numpy as np

from load_verses import load_verses, text_to_embed

HERE = Path(__file__).parent
BATCH = 1000

verses = load_verses()
vectors = np.load(HERE / "vectors.npy")

client = chromadb.PersistentClient(path=str(HERE / "chroma_db")) 
collection = client.get_or_create_collection(
    "quran_verses",
    configuration={"hnsw":{"space": "cosine"}},
)

for start in range(0, len(verses), BATCH):
    batch = verses[start:start + BATCH]
    collection.upsert(
        ids=[v["ref"] for v in batch],
        embeddings=vectors[start:start + BATCH],
        documents=[text_to_embed(v) for v in batch],
        metadatas=batch,
    )
    print(f"{start + len(batch)}/{len(verses)} added")

print(f"Verses in Chroma: {collection.count()}")

