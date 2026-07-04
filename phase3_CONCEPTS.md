# Phase 3 — LangGraph & ReAct Agent Cheat Sheet

## The Agent Loop at a Glance

```
                         START
                           │
                           ▼
                    ┌─────────────┐
              ┌────►│ agent_node  │  ← LLM reasons: "do I need a tool?"
              │     └──────┬──────┘
              │            │
              │      should_continue()
              │            │
              │     ┌──────┴──────┐
              │     │             │
              │  "tools"        "end"
              │     │             │
              │     ▼             ▼
              │  ┌────────┐      END
              │  │  tools │  ← executes requested tool(s)
              │  └────┬───┘
              │       │
              └───────┘
              (loop back with tool results)
```

This is the **ReAct pattern**: Reason → Act → Observe → Reason → ... → Answer

---

## Core LangGraph Concepts

### 1. AgentState — the shared memory
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```
- Every node receives the FULL state and returns partial updates
- `add_messages` is a **reducer**: new messages get APPENDED, not replacing old ones
- Without the reducer, each node's return would overwrite the whole list

```python
# Node returns this:
{"messages": [new_ai_message]}

# LangGraph merges it into state as:
state["messages"] = state["messages"] + [new_ai_message]   # because of add_messages
```

---

### 2. @tool decorator — turning functions into agent abilities
```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search internal documents about RAG, embeddings, vector DBs."""
    # docstring = what the LLM reads to decide WHEN to use this tool
    ...
```
**The docstring IS the interface.** The LLM never sees your code — only the
function name, docstring, and type-hinted parameters. Vague docstrings →
the LLM picks the wrong tool or doesn't use it at all.

---

### 3. bind_tools() — function calling
```python
llm_with_tools = llm.bind_tools([search_knowledge_base, calculator])
response = llm_with_tools.invoke(messages)

# response is either:
#   AIMessage(content="Here's your answer...", tool_calls=[])          # final answer
#   AIMessage(content="", tool_calls=[{"name": "calculator", "args": {...}}])  # wants a tool
```
This relies on the LLM provider supporting native function calling
(Llama 3.1+, GPT-4, Claude all do).

---

### 4. ToolNode — executes requested tools
```python
tool_node = ToolNode([search_knowledge_base, calculator])
```
Internally it:
1. Reads `last_message.tool_calls`
2. Calls each tool with the LLM-provided arguments
3. Wraps each result in a `ToolMessage(content=..., tool_call_id=...)`
4. `tool_call_id` links the result back to the specific request (important for multi-tool calls in one turn)

---

### 5. Conditional Edges — the routing logic
```python
graph.add_conditional_edges(
    "agent",            # from this node
    should_continue,    # this function decides where to go
    {
        "tools": "tools",  # if function returns "tools" -> go to tools node
        "end": END,        # if function returns "end" -> stop
    }
)
```
`should_continue` just checks: does the last AIMessage have `tool_calls`?
- Yes → route to "tools"
- No → route to END (the LLM gave a final text answer)

---

### 6. The Loop-Back Edge
```python
graph.add_edge("tools", "agent")   # ALWAYS go back to agent after tools run
```
This is what makes it a LOOP, not a one-shot pipeline. After tools execute,
the agent sees the tool results (as ToolMessages in state) and reasons again —
maybe it needs ANOTHER tool, or maybe it now has enough info to answer.

---

## Tracing a Real Run

Question: *"What topics are in the KB, and explain FAISS briefly?"*

```
1. agent_node:    LLM sees question -> decides to call list_knowledge_base_topics
2. should_continue: tool_calls present -> route to "tools"
3. tools node:    executes list_knowledge_base_topics() -> returns topic list
4. agent_node:    LLM sees topics -> decides it ALSO needs search_knowledge_base("FAISS")
5. should_continue: tool_calls present -> route to "tools"
6. tools node:    executes search_knowledge_base("FAISS") -> returns FAISS docs
7. agent_node:    LLM has enough info now -> generates final text answer (no tool_calls)
8. should_continue: no tool_calls -> route to END
```
Notice: the agent decided to call TWO tools across TWO loop iterations,
entirely on its own — you never hardcoded "call these 2 tools."

---

## Single-Tool vs Multi-Tool vs No-Tool

| Scenario | What happens |
|----------|---------------|
| "What is RAG?" | 1 loop: search_knowledge_base → answer |
| "Calc 50*3 and explain chunking" | 2 loops: calculator, then search_knowledge_base → answer |
| "Hello, who are you?" | 0 loops: agent answers directly, no tool_calls |

---

## Debugging Tips

```python
# See every step of execution (not just the final answer):
for step in app.stream({"messages": [HumanMessage("...")]}):
    print(step)

# Check what tools were called and in what order:
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        print("Agent requested:", [tc["name"] for tc in msg.tool_calls])
    if isinstance(msg, ToolMessage):
        print("Tool result:", msg.name, "->", msg.content[:100])
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: tool_calls` | Using a model without function-calling support | Use llama-3.1-8b/70b, gpt-4o, or claude-3.5+ |
| Infinite loop (never reaches END) | LLM keeps requesting tools | Add a max iteration guard in state, or improve system prompt |
| `KeyError: 'messages'` | State dict structure mismatch | Always return `{"messages": [...]}`, never raw lists |
| Tool not being called when expected | Weak/vague docstring | Rewrite docstring to be explicit about WHEN to use it |
| `ToolMessage` missing `tool_call_id` | Manual tool node bug | Always pass `tool_call_id=call["id"]` from the original request |

---

## What Carries into Phase 4

In Phase 4, this entire graph (`app = build_agent_graph()`) gets wrapped in a
FastAPI endpoint:

```python
@app_fastapi.post("/chat")
async def chat(question: str):
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    return {"answer": result["messages"][-1].content}
```

And `app.stream()` becomes Server-Sent Events (SSE) so the Streamlit/web UI
shows the agent's reasoning steps live, not just the final answer.
