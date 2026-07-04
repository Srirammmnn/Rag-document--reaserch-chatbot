# Phase 2 — LCEL & RAG Chain Cheat Sheet

## The RAG Pipeline at a Glance

```
User Question
      │
      ▼
┌─────────────────┐
│    Retriever    │  ← vectorstore.as_retriever(k=4)
│  (FAISS search) │  Converts question → embedding → fetches top-k chunks
└────────┬────────┘
         │ List[Document]
         ▼
┌─────────────────┐
│  format_docs()  │  ← Joins chunks into one context string with source headers
└────────┬────────┘
         │ str
         ▼
┌─────────────────────────────────────┐
│         Prompt Template             │  ← Injects {context} + {question}
│  "Answer using ONLY this context:   │
│   {context}                         │
│   Question: {question}"             │
└────────┬────────────────────────────┘
         │ List[ChatMessage]
         ▼
┌─────────────────┐
│      LLM        │  ← Groq / OpenAI / Ollama
│  (Generation)   │  Reads context, generates grounded answer
└────────┬────────┘
         │ AIMessage
         ▼
┌─────────────────┐
│ StrOutputParser │  ← Extracts .content from AIMessage → plain string
└────────┬────────┘
         │ str
         ▼
     Final Answer
```

---

## LCEL — The Pipe Operator

LCEL = LangChain Expression Language. The `|` operator chains Runnables.

```python
# Every component is a Runnable with .invoke(), .stream(), .batch()

chain = retriever | format_docs | prompt | llm | StrOutputParser()
#         │              │           │       │         │
#         ▼              ▼           ▼       ▼         ▼
#    List[Doc]   →    str    →  messages → AIMsg  →  str

# Call it:
answer = chain.invoke("What is RAG?")

# Stream it token by token:
for chunk in chain.stream("What is RAG?"):
    print(chunk, end="", flush=True)

# Batch multiple questions:
answers = chain.batch(["Q1", "Q2", "Q3"])
```

---

## RunnableParallel — Run Two Things at Once

```python
# Run retriever AND pass question through simultaneously
setup = RunnableParallel(
    context = retriever | format_docs,   # branch 1: retrieves + formats
    question = RunnablePassthrough(),    # branch 2: passes input unchanged
)

# Input:  "What is RAG?"
# Output: {"context": "Doc 1: ...\nDoc 2: ...", "question": "What is RAG?"}

chain = setup | prompt | llm | StrOutputParser()
```

---

## RunnablePassthrough — Identity Pass-Through

```python
# Passes input unchanged — used to preserve the original question
# alongside the retrieved context

from langchain_core.runnables import RunnablePassthrough

chain = RunnableParallel(
    context=retriever | format_docs,
    question=RunnablePassthrough(),  # ← question flows through untouched
)
```

---

## Prompt Template — The System Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY this context:\n{context}"),
    ("human", "{question}"),
])

# Fill it manually (for debugging):
filled = prompt.invoke({"context": "RAG stands for...", "question": "What is RAG?"})
print(filled.messages)
# → [SystemMessage("Answer using ONLY..."), HumanMessage("What is RAG?")]
```

---

## ConversationalRAG — Chat History

```python
# MessagesPlaceholder injects the full history into the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Context: {context}"),
    MessagesPlaceholder("chat_history"),   # ← list of HumanMessage + AIMessage
    ("human", "{question}"),
])

# The chain receives a dict with all three:
chain.invoke({
    "question": "Tell me more",
    "chat_history": [
        HumanMessage("What is RAG?"),
        AIMessage("RAG stands for Retrieval-Augmented Generation..."),
    ],
    # context is injected by RunnablePassthrough.assign()
})
```

---

## Retriever Search Types — When to Use Each

| Type | Use When | Trade-off |
|------|----------|-----------|
| `similarity` (default) | Fast, general Q&A | Can return repetitive chunks |
| `mmr` | Need diverse coverage | Slightly slower |
| `similarity_score_threshold` | High-precision, low-noise | May return 0 results |

```python
# similarity (default)
retriever = vs.as_retriever(search_kwargs={"k": 4})

# mmr
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.5}
)

# threshold
retriever = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.7}
)
```

---

## LLM Temperature Guide

| Temperature | Effect | Use For |
|-------------|--------|---------|
| 0.0 | Fully deterministic, consistent | RAG Q&A, fact retrieval |
| 0.3 | Slightly creative | Summaries, explanations |
| 0.7 | More creative | Writing, brainstorming |
| 1.0+ | Very random | Creative writing |

**For RAG: always use temperature=0** — you want grounded, consistent answers.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: vectorstore` | Phase 1 not run | Run `python ingest.py` first |
| `AuthenticationError` | Missing API key | Check `.env` file / export env var |
| `GROQ_API_KEY not found` | .env not loaded | Add `load_dotenv()` at top |
| LLM makes up info (hallucination) | Prompt not strict enough | Add "ONLY use the context" to system prompt |
| Empty `chat_history` error | Wrong type | Use `HumanMessage`/`AIMessage` objects, not strings |
| Slow first response | Model cold start | Normal — subsequent calls are faster |

---

## What Carries into Phase 3

In Phase 3 (LangGraph), the RAG chain becomes a **tool** that the agent can call:

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(question: str) -> str:
    """Search the document knowledge base for information."""
    return answer_chain.invoke(question)

# The LangGraph agent decides WHEN to call this tool
# and can combine it with web search, calculators, etc.
```

The agent loop = decide → tool call → observe → decide → ... → answer
```
