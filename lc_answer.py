import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from answer import SYSTEM, check_citations
from lc_search import search 

llm = ChatAnthropic(model="claude-opus-5-5", max_tokens=16000)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Verses:\n{context}\n\nQuestion: {question}"),
])

chain = prompt | llm

def answer(question):
    docs = search(question)
    verses = [d.metadata for d in docs]
    context = "\n\n".join(f"[{v['ref']}] ({v['surah']}) {v['english']}" for v in verses)

    message = chain.invoke({"context": context, "question": question})
    if message.response_metadata.get("stop_reason") == "refusal":
        return "Claude declined to answer this one.", verses, message.usage_metadata
    return message.text, verses, message.usage_metadata

if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    text, verses, usage = answer(question)
    print(text) 
    print("\nSources:", ", ".join(v["ref"] for v in verses)) 
    cost = usage["input_tokens"] / 1_000_000 *4 + usage["output_tokens"] / 1_000_000 * 20
    print(f"Claude tokens: {usage['input_tokens']} in, {usage['output_tokens']} out (about ${cost:.4f})")
    invented, unused = check_citations(text, verses)
    print("Citations OK" if not invented else f"WARNING, not in sources: {', '.join(sorted(invented))}")
    if unused: 
        print(f"Sources not cited: {', '.join(sorted(unused))}")

