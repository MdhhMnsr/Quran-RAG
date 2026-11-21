import re
import sys
from langchain_anthropic import ChatAnthropic 
from langgraph.graph import StateGraph, MessagesState, START 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError 
from answer import SYSTEM, check_citations 

from langchain_core.tools import tool

from lc_search import search, verses


def format_verse(v):
    return f"[{v['ref']}] ({v['surah']}) {v.get('arabic', '')}\n{v['english']}"

@tool
def search_verses(question: str) -> str:
    """Find the 5 verses most related to a question. Use for meanings, stories and topics."""
    docs = search(question)
    return "\n\n".join(format_verse(d.metadata) for d in docs)

@tool 
def count_word(word: str) -> str: 
    """Count how many times an English word (singular, e.g. 'sun' or 'ant') appears in all 6,236 verses.
    Use for 'how many times' and 'list all' questions."""
    pattern = re.compile(rf"\b{re.escape(word)}s?\b", re.IGNORECASE)
    found = [v for v in verses if pattern.search(v["english"])]
    total = sum(len(pattern.findall(v["english"])) for v in found)
    shown = "\n\n".join(format_verse(v) for v in found[:10])
    return (f"'{word}': {total} times in {len(found)} verses.\n"
            f"All refs: {', '.join('[' + v['ref'] + ']' for v in found)}\n\n"
            f"First 10 verses:\n{shown}")

AGENT_SYSTEM = SYSTEM +"""
Use the tools to get verses. Verses returned by the tools count as provided.
For 'how many' or 'list all' questions, use count_word, once per word if needed."""

tools = [search_verses, count_word]
llm = ChatAnthropic(model="claude-opus-5-5", max_tokens=16000).bind_tools(tools)

def call_claude(state:MessagesState):
    message = llm.invoke([("system", AGENT_SYSTEM)] + state["messages"])
    return {"messages": [message]}

builder = StateGraph(MessagesState)
builder.add_node("claude", call_claude)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "claude")
builder.add_conditional_edges("claude", tools_condition)
builder.add_edge("tools", "claude")
graph = builder.compile()

def ask(question):
    result = graph.invoke(
        {"messages": [("human", question)]},
        {"recursion_limit":10},
    )
    messages = result["messages"]
    final = messages[-1]

    tool_text = "\n".join(m.content for m in messages if m.type == "tool")
    sources = [{"ref":r} for r in set(re.findall(r"\[(\d+:\d+)\]", tool_text))]
    calls = [f"{c['name']}({c['args']})" for m in messages if m.type == "ai" for c in m.tool_calls]
    tokens_in = sum(m.usage_metadata["input_tokens"] for m in messages if m.type == "ai")
    tokens_out = sum(m.usage_metadata["output_tokens"] for m in messages if m.type == "ai")
    return final, sources, calls, tokens_in, tokens_out

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:])
    try: 
        final, sources, calls, tokens_in, tokens_out = ask(question)
    except GraphRecursionError:
        print("Stopped: the agent hit the 10-step limit.")
        sys.exit(1)

    if final.response_metadata.get("stop_reason") == "refusal": 
        print("Claude declined to answer this one.")
    else: 
        print(final.text)
    print("\nTools used:", "; ".join(calls) or "none")
    cost = tokens_in / 1_000_000 * 4 + tokens_out / 1_000_000 *20 
    print(f"Claude tokens: {tokens_in} in, {tokens_out} out (about ${cost:.4f})")
    invented, unused = check_citations(final.text, sources)
    print("Citations OK" if not invented else f"WARNING, not in sources: {', '.join(sorted(invented))}")
    
