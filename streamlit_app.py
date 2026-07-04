"""
Phase 4: Streamlit UI — Animated RAG Chatbot
=============================================
"""
import json, uuid, time, requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Agent", page_icon="🤖", layout="wide")

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* ── Header banner ── */
.rag-header {
    background: linear-gradient(90deg, #667eea, #764ba2, #f64f59);
    background-size: 300% 300%;
    animation: gradientShift 4s ease infinite;
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
}
.rag-header h1 {
    color: #fff !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Chat containers ── */
.user-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    margin: 10px 0 10px 60px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35);
    animation: slideInRight 0.3s ease;
    line-height: 1.6;
}
.bot-bubble {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    color: #f0f0ff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 18px 18px 18px 4px;
    padding: 16px 22px;
    margin: 10px 60px 10px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    animation: slideInLeft 0.35s ease;
    line-height: 1.75;
}
.bot-bubble strong, .bot-bubble b {
    color: #a78bfa !important;
    font-weight: 800 !important;
}
.bot-bubble code {
    background: rgba(167,139,250,0.2);
    color: #c4b5fd;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
}
.user-label {
    text-align: right;
    font-size: 0.72rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.bot-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
@keyframes slideInRight {
    from { opacity:0; transform: translateX(30px); }
    to   { opacity:1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity:0; transform: translateX(-30px); }
    to   { opacity:1; transform: translateX(0); }
}

/* ── Typing dots ── */
.typing-indicator {
    display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 22px;
    margin: 10px 60px 10px 0;
    width: fit-content;
}
.dot {
    width: 9px; height: 9px;
    background: #a78bfa;
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay: 0.2s; background: #60a5fa; }
.dot:nth-child(3) { animation-delay: 0.4s; background: #f472b6; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
    30%            { transform: translateY(-10px); opacity: 1; }
}

/* ── Sources badge ── */
.source-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.4);
    color: #c4b5fd !important;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 6px 4px 0 0;
}

/* ── Route badge ── */
.route-pill {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.35);
    color: #93c5fd !important;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 20px;
    padding: 2px 10px;
    margin-bottom: 8px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 2px solid rgba(167,139,250,0.4) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: rgba(255,255,255,0.4);
}
.empty-state .icon { font-size: 4rem; margin-bottom: 16px; animation: pulse 2s infinite; }
.empty-state h3 { font-size: 1.4rem; font-weight: 700; color: rgba(255,255,255,0.6) !important; }
.empty-state p  { font-size: 0.95rem; font-weight: 500; }
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50%       { transform: scale(1.1); opacity: 1; }
}

/* Hide default streamlit chat avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }
[data-testid="stChatMessage"] { background: transparent !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Agent")
    st.markdown("---")

    # Health
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health["vectorstore_loaded"]:
            st.success("☁️ Pinecone Connected")
        else:
            st.warning("⚠️ No documents yet")
    except:
        st.error("❌ Backend offline")
        st.code("uvicorn main:app --reload --port 8000")
        st.stop()

    # Sources
    try:
        src_data = requests.get(f"{API_URL}/sources", timeout=5).json()
        if src_data["sources"]:
            with st.expander(f"📚 {len(src_data['sources'])} document(s) indexed", expanded=False):
                st.caption(f"{src_data['total_chunks']} chunks total")
                for s in src_data["sources"]:
                    st.markdown(f"📄 `{s}`", unsafe_allow_html=True)
        else:
            st.info("No documents indexed yet")
    except:
        pass

    st.markdown("---")

    # Upload
    st.markdown("**📥 Upload Document**")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        if st.button("⚡ Ingest Now", type="primary", use_container_width=True):
            with st.spinner(f"Processing {uploaded.name}..."):
                try:
                    r = requests.post(f"{API_URL}/ingest",
                                      files={"file": (uploaded.name, uploaded.getvalue())},
                                      timeout=180)
                    if r.status_code == 200:
                        res = r.json()
                        st.success(f"✅ {res['chunks_added']} chunks added!")
                        st.caption(res.get("status", ""))
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", "Ingest failed"))
                except Exception as e:
                    st.error(str(e))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        streaming_on = st.toggle("Stream", value=True)
    with col2:
        show_route = st.toggle("Routing", value=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
        except:
            pass
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Route Legend**
    - 🗂️ `rag` — Your documents
    - 🌐 `web` — Web search
    - 🧮 `math` — Calculation
    - 🐍 `python` — Code
    - 🧠 `llm` — General AI
    """)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <h1>🤖 RAG Intelligence Agent</h1>
  <p>Hybrid Search · CrossEncoder Reranking · LangGraph Router · Semantic Cache</p>
</div>
""", unsafe_allow_html=True)

# ─── ROUTE ICON MAP ───────────────────────────────────────────────────────────
ROUTE_ICONS = {
    "routed_to_rag": ("🗂️", "rag"),
    "routed_to_web": ("🌐", "web"),
    "routed_to_math": ("🧮", "math"),
    "routed_to_python": ("🐍", "python"),
    "routed_to_llm": ("🧠", "llm"),
}

def render_route_pill(tool_calls):
    if not tool_calls:
        return ""
    for tc in tool_calls:
        if tc in ROUTE_ICONS:
            icon, label = ROUTE_ICONS[tc]
            return f'<span class="route-pill">{icon} routed → {label}</span>'
    return ""

# ─── RENDER HISTORY ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">💬</div>
      <h3>Start a conversation</h3>
      <p>Upload a document in the sidebar, then ask me anything about it.<br>
      Try: <em>"What are my certifications?"</em> or <em>"Summarise the document"</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-label">You</div><div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            pill = render_route_pill(msg.get("tool_calls", [])) if show_route else ""
            sources_html = "".join(f'<span class="source-badge">📄 {s}</span>' for s in msg.get("sources", []))
            st.markdown(
                f'<div class="bot-label">🤖 Agent</div>'
                f'{pill}'
                f'<div class="bot-bubble">{msg["content"]}{("<br><br>" + sources_html) if sources_html else ""}</div>',
                unsafe_allow_html=True
            )

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def call_blocking(question: str) -> dict:
    r = requests.post(f"{API_URL}/chat",
                      json={"question": question, "session_id": st.session_state.session_id},
                      timeout=90)
    r.raise_for_status()
    return r.json()

def stream_response(question: str, typing_slot, answer_slot):
    r = requests.post(f"{API_URL}/chat/stream",
                      json={"question": question, "session_id": st.session_state.session_id},
                      stream=True, timeout=120)
    r.raise_for_status()

    answer, tool_calls, sources = "", [], set()

    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        ev = json.loads(raw[6:])
        t = ev.get("type")

        if t == "tool_call":
            tool_calls.append(ev["tool"])
            icon, label = ROUTE_ICONS.get(ev["tool"], ("⚙️", ev["tool"]))
            typing_slot.markdown(f"""
            <div class="typing-indicator">
              <span style="color:#a78bfa;font-weight:700;font-size:.85rem">{icon} Routing → {label}</span>
              <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>""", unsafe_allow_html=True)

        elif t == "tool_result":
            typing_slot.markdown("""
            <div class="typing-indicator">
              <span style="color:#60a5fa;font-weight:700;font-size:.85rem">⚡ Processing results</span>
              <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>""", unsafe_allow_html=True)

        elif t == "answer":
            answer = ev["content"]
            typing_slot.empty()
            pill = render_route_pill(tool_calls) if show_route else ""
            sources_html = "".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
            answer_slot.markdown(
                f'{pill}<div class="bot-bubble">{answer}{("<br><br>" + sources_html) if sources_html else ""}</div>',
                unsafe_allow_html=True
            )

        elif t == "done":
            typing_slot.empty()

        elif t == "error":
            typing_slot.empty()
            answer_slot.error(f"Error: {ev['message']}")

    return answer, tool_calls, list(sources)

# ─── CHAT INPUT ───────────────────────────────────────────────────────────────
if question := st.chat_input("Ask anything about your documents..."):
    # Render user bubble
    st.markdown(f'<div class="user-label">You</div><div class="user-bubble">{question}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})

    st.markdown('<div class="bot-label">🤖 Agent</div>', unsafe_allow_html=True)
    typing_slot  = st.empty()
    answer_slot  = st.empty()

    # Show typing animation immediately
    typing_slot.markdown("""
    <div class="typing-indicator">
      <span style="color:#a78bfa;font-weight:700;font-size:.85rem">Thinking</span>
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>""", unsafe_allow_html=True)

    try:
        if streaming_on:
            answer, tool_calls, sources = stream_response(question, typing_slot, answer_slot)
        else:
            result = call_blocking(question)
            answer = result["answer"]
            tool_calls = result.get("tool_calls_made", [])
            sources = result.get("sources", [])
            typing_slot.empty()
            pill = render_route_pill(tool_calls) if show_route else ""
            sources_html = "".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
            answer_slot.markdown(
                f'{pill}<div class="bot-bubble">{answer}{("<br><br>" + sources_html) if sources_html else ""}</div>',
                unsafe_allow_html=True
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "tool_calls": tool_calls,
        })

    except Exception as e:
        typing_slot.empty()
        answer_slot.error(f"❌ {e}")
