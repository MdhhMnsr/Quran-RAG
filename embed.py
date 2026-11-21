from ctypes.wintypes import SIZE
import time
from pathlib import Path

import numpy as np
import voyageai

from load_verses import load_verses, text_to_embed, PRICE_PER_MILLION

HERE = Path(__file__).parent
SAVED = HERE / "vectors.npy"
TEMP = HERE / "vectors_temp.npy"

MODEL = "voyage-4"
SIZE = 1024
MAX_CHARS = 24_000
WAIT_SECONDS = 65
MAX_BATCHES = None

def next_batch(verses,start):
    texts = []
    chars = 0 
    for verse in verses[start:]:
        text = text_to_embed(verse)
        if texts and chars + len(text) > MAX_CHARS: 
            break 
        texts.append(text)
        chars += len(text) 
    return texts

verses = load_verses()
vo = voyageai.Client(max_retries=3)

if SAVED.exists():
    vectors = np.load(SAVED)
    print(f"Found{len(vectors)} verses already embedded" )
else:
    vectors =np.empty((0,SIZE), dtype=np.float32)

tokens_used = 0 
batches = 0

while len(vectors) < len(verses):
    if MAX_BATCHES is not None and batches == MAX_BATCHES:
        break
    if batches > 0: 
        time.sleep(WAIT_SECONDS)

    texts = next_batch(verses,len(vectors))
    result = vo.embed(texts,model=MODEL, input_type="document")

    new = np.array(result.embeddings, dtype=np.float32)
    vectors = np.vstack([vectors,new])
    np.save(TEMP, vectors)
    TEMP.replace(SAVED)

    tokens_used += result.total_tokens
    batches += 1
    print(f"Batch {batches}: {len(texts)} verses, {result.total_tokens} tokens,{len(vectors)}/{len(verses)} done")

print(f"Tokens this run :{tokens_used:,}")
print(f"Value at list price :${tokens_used / 1_000_000 * PRICE_PER_MILLION:04f}")

