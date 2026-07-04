import sys
import asyncio
from langchain_core.messages import HumanMessage
from agent import build_agent_graph

graph = build_agent_graph()

queries = [
    ("RAG", "What are my certifications?"),
    ("MATH", "What is 120 * 45?"),
    ("PYTHON", "Write a python script to sort the list [3, 1, 4, 1, 5]"),
    ("WEB", "What is the stock price of Apple right now?"),
    ("LLM", "Tell me a very short joke about a programmer.")
]

for name, q in queries:
    print(f"\n{'='*50}\nTESTING NODE: {name}\nQUESTION: {q}")
    try:
        res = graph.invoke({"messages": [HumanMessage(content=q)]})
        print(f"RESPONSE:\n{res['messages'][-1].content}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
