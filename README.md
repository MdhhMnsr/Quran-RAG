# Quran RAG: ask a question, get an answer from the verses

Ask a question about the Quran in English, Malay or Arabic. The system finds the most relevant verses out of all 6,236, and Claude answers **only from those verses**, citing each one as `[surah:ayah]`. A check at the end flags any citation that wasn't among the verses it was given.

I built this with two intentions:

1. To answer my own questions about the Quran from what the verses actually say, not from what people have said about them over time.
2. To understand RAG properly, with room to add to it later: first in plain Python, then again in LangChain, LangGraph and LangSmith, measuring each version against the same questions.

The verse data comes from my own project, [Connect The Meaning](https://connectthemeaning.netlify.app).

## How it works

```
question
  -> embed the question (Voyage voyage-4)
  -> vector search over 6,236 verses, keep the top 30
  -> rerank the 30 (Voyage rerank-2.5), keep the top 5
  -> Claude (claude-opus-5-5) answers from those 5 verses only
  -> citation check: every [surah:ayah] cited must be one of the 5
```

Each verse is embedded once, as its English and Malay together, and saved, so a question costs only one small embedding call.

## Results

**Hit rate @5**: does at least one expected verse appear in the top 5?

12 hand-written questions (the first test set):

| Version | Vector only | + Reranker |
|---|---|---|
| Plain Python + NumPy | 8/12 | 9/12 |
| LangChain + NumPy | 8/12 | 9/12 |
| LangChain + Chroma | 8/12 | 9/12 |

50 questions (topics, stories, people, counts, rules; English, Malay, Arabic), reranked, run as LangSmith experiments:

| Version | Hit rate |
|---|---|
| English + Malay vectors | 45/50 (90%) |
| + separate Arabic vectors | 45/50 (90%) |

The Arabic vectors fixed one Arabic miss (the elephant question now finds 105:1 first) but pushed one Malay question's verse from rank 29 to 32, just outside the 30 sent to the reranker.

**Caveat:** I wrote 38 of the 50 questions after building the system, and many of them use words that appear in the verses, so 90% probably overstates real-world performance. The original 12 are the stricter test.

## What I learned

- **The reranker fixed what vectors alone missed.** For "rules about inheritance", vector search ranked a verse that only contains the word *inheritors* first. The reranker read each question and verse together and moved the actual rules (4:11, 4:12, 4:176) to the top.
- **Some misses came from my answer key, not the search.** "How many times is the sun mentioned?" returned five verses that all mention the sun, but my key accepted only 91:1. I kept the original key for scoring and noted this instead of relabelling after seeing the results.
- **Search can't count.** Top-5 search can't answer "how many times" or "name all". In the LangGraph version I gave the agent a second tool, `count_word`, which scans every verse. It found the sun in 33 verses.
- **Some answers aren't in the text.** "What was the first verse revealed?" (96:1) is a real miss. That fact comes from hadith and tafsir, not from the verse itself, so a search over the verse text can't find it.
- **The vector store changes results even when the score doesn't.** Chroma's approximate index gave different top-30 lists from exact NumPy search on some questions (22 of 30 overlapping for "spiders"), but the hit rate didn't change. A single number can hide differences like that.

## Files

| File | What it does |
|---|---|
| `load_verses.py` | Loads the 6,236 verses (English, Malay, Arabic) |
| `embed.py`, `embed_ar.py` | Embed every verse once and save the vectors (English + Malay, and Arabic) |
| `search.py` | Plain Python: vector search + reranker |
| `answer.py` | Plain Python: Claude answers from the retrieved verses, with the citation check |
| `eval.py` | Plain Python: hit rate with and without the reranker |
| `lc_search.py`, `lc_answer.py`, `lc_eval.py` | The same pipeline in LangChain, with a NumPy or Chroma vector store |
| `chroma_build.py` | Loads the saved vectors into Chroma (no re-embedding) |
| `lg_agent.py` | LangGraph agent with two tools: `search_verses` and `count_word` |
| `ls_eval.py` | The eval as a LangSmith dataset and experiment |
| `questions.txt` | The test questions and their expected verses |

## Run it yourself

The verse text and the vectors are **not** in this repo, because the English translation (Saheeh International) isn't mine to publish. To run the code you need the verse files in the Connect The Meaning JSON format.

1. Python 3.13 or 3.14 (the LangChain packages don't install on 3.15 yet).
2. Install the packages:
   ```bash
   pip install anthropic voyageai numpy langchain-core langchain-anthropic langchain-voyageai langchain-chroma langgraph langsmith
   ```
3. Set your keys as environment variables: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, and optionally `LANGSMITH_API_KEY`. Set `CTM_DATA` to the folder that holds the verse files.
4. Embed the verses once (about 650K tokens, which fits inside Voyage's free allowance):
   ```bash
   python embed.py
   ```
5. Ask a question:
   ```bash
   python answer.py what does the Quran say about patience
   ```

## Cost

| Step | Cost |
|---|---|
| Embedding all verses (once) | ~$0.04 at list price, free within Voyage's free tokens |
| One answered question | ~$0.01–0.03 (mostly Claude) |
| One agent question (LangGraph) | ~$0.02–0.20, depending on how many tools it calls |
| A full 50-question eval (search only, no Claude) | a few cents |

## Arabic search: harakat removed for embedding only

Arabic questions are matched against a separate set of Arabic vectors (`vectors_ar.npy`). Before embedding, the harakat (vowel marks such as fathah, kasrah, dammah, sukun, shaddah and tanwin) and the tatweel are removed, and alef wasla (ٱ) is written as a plain alef (ا).

Example: `بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ` is embedded as `بسم الله الرحمن الرحيم`.

> **Note:** this matches the way Arabic questions are usually typed, without harakat, so a question and its verse look alike to the search. It also cuts the Arabic from about 636K to about 172K embedding tokens (164,600 counted by Voyage for the real run).

The original text is not changed:
- The stripped form is used only to build the search vectors.
- Answers quote the full Arabic with harakat, exactly as stored in the Connect The Meaning data.
- The live Connect The Meaning site and its data files are not touched.

## Limits

- Claude answers only from the verses search gives it. It doesn't use tafsir or hadith, and it doesn't give religious rulings.
- Answers are for learning and exploring, not scholarship. Always check the verses themselves.
