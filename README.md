# 🧠 Rag-Document-Research-Chatbot

An advanced, agentic Retrieval-Augmented Generation (RAG) system with a modern React frontend and a LangGraph-powered FastAPI backend. The system intelligently routes user queries between local document retrieval, real-time web searches, Python execution, math evaluation, and general LLM conversation.

---

## 🌟 Key Features

* **Intelligent Routing (LangGraph):** Uses `llama-3.1-8b-instant` to semantically route queries to 5 distinct specialized nodes (RAG, Math, Web, Python, General Chat).
* **Hybrid Search Retrieval:** Combines Dense Vectors (Pinecone + HuggingFace) and Sparse Vectors (Local BM25) for high-recall document fetching.
* **CrossEncoder Reranking:** Re-ranks the retrieved hybrid chunks to guarantee pinpoint accuracy before passing context to the LLM.
* **Real-time Streaming:** The React frontend consumes Server-Sent Events (SSE) to render thoughts and responses word-by-word instantly.
* **Semantic Query Caching:** Bypasses LLM generation for repeated questions to save API costs and return answers in milliseconds.

## 🛠️ Technology Stack

**Frontend:**
* React 18 (Vite)
* Framer Motion (Animations)
* Lucide React (Icons)
* React Markdown

**Backend:**
* FastAPI (Python web server)
* LangGraph / LangChain (Agentic logic)
* Groq Cloud (`llama-3.1-8b-instant`)
* Pinecone (Vector Database)
* SentenceTransformers (HuggingFace local embeddings)

---

## 🚀 Local Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Srirammmnn/Rag-document--reaserch-chatbot.git
cd Rag-document--reaserch-chatbot
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY="your_groq_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="your_pinecone_index_name"
```

### 3. Start the Backend (FastAPI)
Install Python dependencies and start the Uvicorn server:
```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### 4. Start the Frontend (React)
Open a new terminal window, navigate to the frontend folder, and start Vite:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to chat with the agent!

---

## ☁️ Deployment

This repository is optimized for deployment on **Render** (Backend) and **Vercel** (Frontend). See `DEPLOYMENT.md` for full instructions.
