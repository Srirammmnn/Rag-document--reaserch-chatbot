"""
Phase 2: RAG Chain — Retrieval + Generation
============================================
Flow: Query → Retriever (FAISS) → Context Docs → Prompt Template → LLM → Answer + Sources

Builds directly on Phase 1 (ingest.py must be run first to create the vectorstore).

Key concepts covered:
  - LCEL (LangChain Expression Language) — pipe operator chaining
  - Retriever — wrapping vectorstore for RAG
  - PromptTemplate — structured prompts with variables
  - LLM integration (Groq = free, fast | OpenAI | Ollama = local/offline)
  - Runnable chain composition
  - Chat history / multi-turn memory
  - Source citation
  - Streaming responses
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Core LangChain
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# Vector store + embeddings (from Phase 1)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLM options — uncomment the one you want to use
# Option A: Groq (FREE, fastest, recommended for learning)
from langchain_groq import ChatGroq

# Option B: OpenAI (paid, most capable)
# from langchain_openai import ChatOpenAI

# Option C: Ollama (100% local/offline, no API key)
# from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY / OPENAI_API_KEY from .env file


# ─────────────────────────────────────────────
# STEP 1: LLM SETUP
# ─────────────────────────────────────────────

def get_llm(provider: str = "groq"):
    """
    Choose your LLM provider.

    GROQ (recommended for learning):
      - Free tier: 30 req/min, 6000 tokens/min
      - Get API key: https://console.groq.com  (takes 30 seconds)
      - Models: llama-3.1-8b-instant (fast), llama-3.1-70b-versatile (smart)
      - Set env: GROQ_API_KEY=your_key

    OPENAI:
      - Paid but most capable
      - Get API key: https://platform.openai.com
      - Set env: OPENAI_API_KEY=your_key

    OLLAMA (local/offline, no API key needed):
      - Install: https://ollama.com/download
      - Pull model: `ollama pull llama3.2`
      - Run: `ollama serve` (starts local server)
      - Zero cost, full privacy
    """
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("  🚀 Using Google Gemini (gemini-1.5-flash) — free & flawless tool calling")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=1024,
        )
    elif provider == "groq":
        print("  🚀 Using Groq LLM (llama-3.1-8b-instant) — free & fast")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1024,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        print("  🤖 Using OpenAI GPT-4o-mini")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1024,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        print("  🏠 Using Ollama (local, offline)")
        return ChatOllama(
            model="llama3.2",
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'gemini', 'groq', 'openai', or 'ollama'")


# ─────────────────────────────────────────────
# STEP 2: LOAD VECTORSTORE FROM PHASE 1
# ─────────────────────────────────────────────

def load_vectorstore(vectorstore_path: str = "vectorstore") -> FAISS:
    """
    Load the FAISS index built in Phase 1.
    The embedding model MUST be the same one used during ingestion.
    """
    if not Path(vectorstore_path).exists():
        raise FileNotFoundError(
            f"Vector store not found at '{vectorstore_path}/'.\n"
            "Run Phase 1 first: python ingest.py"
        )

    print(f"  📂 Loading FAISS vector store from '{vectorstore_path}/'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",     # MUST match Phase 1
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"  ✅ Loaded {vectorstore.index.ntotal} vectors")
    return vectorstore


# ─────────────────────────────────────────────
# STEP 3: CREATE THE RETRIEVER
# ─────────────────────────────────────────────

def get_retriever(vectorstore: FAISS, search_type: str = "similarity", k: int = 4):
    """
    Wrap the vectorstore as a LangChain Retriever.

    The retriever is the "R" in RAG — it fetches relevant context chunks
    given a query string.

    search_type options:
      "similarity"            → top-k by cosine/L2 distance (default, fast)
      "mmr"                   → Maximal Marginal Relevance (diverse results)
      "similarity_score_threshold" → only return chunks above a score threshold

    k: how many chunks to retrieve
       More chunks = more context = better coverage BUT larger prompt = higher cost
       Typical: 3–5 for focused Q&A, 6–8 for complex research questions
    """
    print(f"  🔍 Creating retriever: type={search_type}, k={k}")

    if search_type == "similarity":
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    elif search_type == "mmr":
        # MMR balances relevance + diversity
        # Good when chunks can be repetitive
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,   # fetch 3x candidates, then pick k diverse ones
                "lambda_mult": 0.5, # 0=max diversity, 1=max relevance
            },
        )
    elif search_type == "threshold":
        # Only return chunks that are VERY similar to the query
        # Returns fewer results but higher precision
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.7},
        )

    return retriever


# ─────────────────────────────────────────────
# STEP 4: PROMPT TEMPLATES
# ─────────────────────────────────────────────

def get_qa_prompt() -> ChatPromptTemplate:
    """
    System prompt for RAG Q&A.

    The prompt has two key variables:
      {context}  → injected retrieved chunks (the "augmented" part of RAG)
      {question} → the user's query

    Prompt engineering tips for RAG:
      1. Tell the LLM to ONLY use the provided context (prevents hallucination)
      2. Tell it to say "I don't know" if context is insufficient
      3. Ask it to cite sources (optional but great for credibility)
      4. Be specific about output format
    """
    system_template = """You are a precise, helpful AI research assistant.
Your job is to answer questions based ONLY on the provided context documents.

Rules:
1. Use ONLY information from the context below — never your training data
2. If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to answer this."
3. Always cite the source document(s) at the end of your answer
4. Be concise but complete — no unnecessary padding
5. If asked for a list, use bullet points

Context Documents:
-------------------
{context}
-------------------

Answer the question based on the context above."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}"),
    ])
    return prompt


def get_conversational_prompt() -> ChatPromptTemplate:
    """
    Prompt template for multi-turn conversation with memory.

    MessagesPlaceholder("chat_history") injects the full conversation
    history so the LLM can understand follow-up questions like
    "tell me more about that" or "what did you mean by X?"

    This is what separates a chatbot from a one-shot Q&A system.
    """
    system_template = """You are a helpful AI research assistant with access to a knowledge base.
Answer questions using ONLY the provided context. Maintain conversation history for follow-up questions.
If the context doesn't answer the question, say so clearly.

Context from knowledge base:
-------------------
{context}
-------------------"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history"),  # ← injects chat history here
        ("human", "{question}"),
    ])
    return prompt


# ─────────────────────────────────────────────
# STEP 5: FORMAT RETRIEVED DOCS
# ─────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved Document chunks into a single context string for the prompt.

    This is a critical but often overlooked step.
    The LLM sees this exact string as its "context" — format it clearly.

    We include the source metadata so the LLM can cite it.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        topic = doc.metadata.get("topic", "")
        page = doc.metadata.get("page", "")

        header = f"[Document {i}] Source: {source}"
        if topic:
            header += f" | Topic: {topic}"
        if page:
            header += f" | Page: {page}"

        formatted.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted)


def get_sources(docs: List[Document]) -> List[str]:
    """Extract unique source names from retrieved docs for display."""
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            sources.append(src)
            seen.add(src)
    return sources


# ─────────────────────────────────────────────
# STEP 6: BUILD THE RAG CHAIN (LCEL)
# ─────────────────────────────────────────────

def build_rag_chain(retriever, llm, prompt) -> dict:
    """
    Build the RAG chain using LCEL (LangChain Expression Language).

    LCEL uses the pipe operator | to chain Runnables together.
    Each step's output becomes the next step's input.

    Chain breakdown:

    RunnableParallel runs TWO things simultaneously when you call chain.invoke():
      1. "context"  → runs retriever on the question, formats the docs
      2. "question" → passes the question straight through (RunnablePassthrough)

    Then the combined dict {context, question} flows into:
      → prompt  (formats context + question into messages)
      → llm     (generates the answer)
      → parser  (extracts the string from the AIMessage)

    LCEL visual:
      question
        ├─→ retriever → format_docs ─→ "context" ─┐
        └─────────────────────────────→ "question" ─┤
                                                    ↓
                                               prompt template
                                                    ↓
                                                   llm
                                                    ↓
                                             StrOutputParser
                                                    ↓
                                               answer string
    """
    # Step 1: Retrieve docs AND pass question through simultaneously
    retrieval_chain = RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough(),
        # We also capture raw docs for source display:
        docs=retriever,
    )

    # Step 2: Full pipeline
    # Note: prompt only needs {context} and {question}, not {docs}
    # We split the chain to capture docs separately

    # Clean chain for LLM answer
    answer_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chain that also returns source documents
    chain_with_sources = RunnableParallel(
        answer=answer_chain,
        docs=retriever,
    )

    return {
        "answer_chain": answer_chain,            # simple: input=str, output=str
        "chain_with_sources": chain_with_sources,  # input=str, output={answer, docs}
    }


# ─────────────────────────────────────────────
# STEP 7: CONVERSATIONAL RAG WITH MEMORY
# ─────────────────────────────────────────────

class ConversationalRAG:
    """
    Multi-turn RAG chatbot with chat history.

    Maintains a running conversation history so follow-up questions
    like "tell me more" or "what about X?" work correctly.

    The history is passed into the prompt via MessagesPlaceholder,
    giving the LLM full context of the conversation.

    Usage:
        bot = ConversationalRAG(retriever, llm)
        answer1 = bot.chat("What is RAG?")
        answer2 = bot.chat("Can you give an example?")  # knows context from Q1
    """

    def __init__(self, retriever, llm, max_history: int = 10):
        self.retriever = retriever
        self.llm = llm
        self.chat_history: List = []  # List of HumanMessage / AIMessage
        self.max_history = max_history
        self.prompt = get_conversational_prompt()

        # Build the conversational chain
        # RunnablePassthrough.assign() adds new keys to the running dict
        self.chain = (
            RunnablePassthrough.assign(
                # Retrieve docs based on current question
                context=RunnableLambda(
                    lambda x: format_docs(retriever.invoke(x["question"]))
                ),
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, question: str, verbose: bool = True) -> str:
        """
        Ask a question. Automatically includes chat history.
        """
        if verbose:
            print(f"\n👤 You: {question}")

        # Invoke chain with question + history
        answer = self.chain.invoke({
            "question": question,
            "chat_history": self.chat_history,
        })

        # Update history (keep last N turns to manage token count)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2:]

        if verbose:
            print(f"\n🤖 Assistant: {answer}")

        return answer

    def get_retrieved_sources(self, question: str) -> List[str]:
        """See which documents were retrieved for a question."""
        docs = self.retriever.invoke(question)
        return get_sources(docs)

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []
        print("  🔄 Chat history cleared")


# ─────────────────────────────────────────────
# STEP 8: STREAMING RESPONSE
# ─────────────────────────────────────────────

def stream_answer(chain, question: str) -> str:
    """
    Stream the LLM response token by token.
    Makes the UI feel responsive — user sees text appearing in real-time.
    Used in Phase 4 with FastAPI SSE.
    """
    print(f"\n👤 Question: {question}")
    print("\n🤖 Answer (streaming): ", end="", flush=True)

    full_answer = ""
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
        full_answer += chunk

    print()  # newline after streaming
    return full_answer


# ─────────────────────────────────────────────
# STEP 9: ANSWER WITH SOURCES
# ─────────────────────────────────────────────

def ask_with_sources(chain_with_sources, question: str) -> Dict[str, Any]:
    """
    Ask a question and get both the answer AND the source documents.
    This is what you'll expose in the API in Phase 4.

    Returns:
        {
            "question": str,
            "answer": str,
            "sources": List[str],
            "retrieved_docs": List[Document]
        }
    """
    result = chain_with_sources.invoke(question)

    answer = result["answer"]
    docs = result["docs"]
    sources = get_sources(docs)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_docs": docs,
    }


# ─────────────────────────────────────────────
# STEP 10: EVALUATION — DID RAG HELP?
# ─────────────────────────────────────────────

def evaluate_retrieval(retriever, test_questions: List[str]) -> None:
    """
    Manual retrieval evaluation — inspect what gets retrieved.
    Before adding RAGAS (Phase 5), do this visual sanity check:

    For each question, look at the retrieved chunks and ask:
      - Are they actually relevant?
      - Is the key information present?
      - Are chunks too long/short?

    This tells you if you need to tune chunk_size, k, or search_type.
    """
    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION")
    print("=" * 60)

    for q in test_questions:
        docs = retriever.invoke(q)
        print(f"\n❓ Question: {q}")
        print(f"   Retrieved {len(docs)} chunks:")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "?")
            preview = doc.page_content.strip()[:100].replace("\n", " ")
            print(f"   [{i}] {src}: {preview}...")
        print()


# ─────────────────────────────────────────────
# MAIN: RUN THE FULL RAG PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 2: RAG Chain")
    print("=" * 60)

    # ── 1. LOAD VECTORSTORE (from Phase 1) ──────────────
    print("\n📥 STEP 1: Loading vector store")
    vectorstore = load_vectorstore("vectorstore")

    # ── 2. SET UP LLM ───────────────────────────────────
    print("\n🤖 STEP 2: Loading LLM")
    # Change provider to "openai" or "ollama" if needed
    # For groq: get free key at https://console.groq.com
    llm = get_llm(provider="groq")

    # ── 3. CREATE RETRIEVER ─────────────────────────────
    print("\n🔍 STEP 3: Creating retriever")
    retriever = get_retriever(vectorstore, search_type="mmr", k=4)

    # ── 4. BUILD CHAINS ─────────────────────────────────
    print("\n⛓️  STEP 4: Building RAG chains")
    prompt = get_qa_prompt()
    chains = build_rag_chain(retriever, llm, prompt)
    answer_chain = chains["answer_chain"]
    chain_with_sources = chains["chain_with_sources"]
    print("   ✅ Chains built")

    # ── 5. RETRIEVAL EVAL (before LLM) ─────────────────
    evaluate_retrieval(retriever, [
        "How does RAG work?",
        "What are vector embeddings?",
        "How does FAISS search work?",
    ])

    # ── 6. SIMPLE Q&A ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 1: Simple Q&A")
    print("=" * 60)

    question = "What is RAG and why is it useful?"
    print(f"\n❓ Question: {question}")
    answer = answer_chain.invoke(question)
    print(f"\n💬 Answer:\n{answer}")

    # ── 7. Q&A WITH SOURCES ─────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 2: Answer with Sources")
    print("=" * 60)

    result = ask_with_sources(chain_with_sources, "How does FAISS similarity search work?")
    print(f"\n❓ Question: {result['question']}")
    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n📚 Sources used: {result['sources']}")

    # ── 8. STREAMING ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 3: Streaming Response")
    print("=" * 60)
    stream_answer(answer_chain, "Explain text chunking and why chunk overlap matters.")

    # ── 9. CONVERSATIONAL RAG ───────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 4: Multi-turn Conversation")
    print("=" * 60)

    bot = ConversationalRAG(retriever, llm)

    # Turn 1
    bot.chat("What is LangGraph and what is it used for?")

    # Turn 2 — follow-up question, relies on conversation history
    bot.chat("What are the key components you just mentioned?")

    # Turn 3 — another follow-up
    bot.chat("How is that different from a regular LangChain chain?")

    # Show history length
    print(f"\n   📝 Chat history: {len(bot.chat_history) // 2} turns stored")

    print("\n" + "=" * 60)
    print("  ✅ Phase 2 Complete!")
    print("  Next: Phase 3 — LangGraph Agentic Loop")
    print("=" * 60)


if __name__ == "__main__":
    main()
