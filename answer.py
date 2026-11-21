import sys 
import re

import anthropic

from search import search

MODEL = "claude-opus-5-5"
client = anthropic.Anthropic()

SYSTEM = """You answer questions about the Quran using only the verses provided.
            Cite every point with its reference in brackets, e.g.[2:153]. 
            If the verses do not answer the question, say so plainly. Do not add verses that were not provided. 
            Do not give religious rulings; describe what the verse say."""

def answer(question):
    verses, search_tokens = search(question)
    context = "\n\n".join(f"[{v['ref']}] ({v['surah']}) {v['english']}" for v in verses)

    response = client.messages.create(
        model=MODEL, 
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Verses:\n{context}\n\nQuestion: {question}"}],  
    )
    if response.stop_reason == 'refusal':
        return "Claude declined to answer this one.", verses, response.usage 
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, verses, response.usage 



def check_citations(text,verses):
    cited = set(re.findall(r"\[(\d+:\d+)\]", text))
    given = {v["ref"] for v in verses}
    return cited - given, given - cited

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) 
    text,verses, usage = answer(question)
    print(text)
    print("\nSources:", ", ".join(v["ref"] for v in verses)) 
    cost = usage.input_tokens / 1_000_000 * 4 + usage.output_tokens / 1_000_000 *20 
    print(f"Claude tokens: {usage.input_tokens} in, {usage.output_tokens} out (about ${cost:.4f})")
    invented, unused = check_citations(text, verses)
    print("Citations OK" if not invented else f"WARNING, not in sources: {','.join(sorted(invented))}")
    if unused:
        print(f"Sources not cited: {', '.join(sorted(unused))}")