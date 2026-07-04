# NeuRAG: Advanced Agentic Retrieval System
**Complete System Architecture & Deployment Guide**

---

## 1. System Architecture Overview

NeuRAG is built on a modern, decoupled full-stack architecture that emphasizes real-time streaming, high-speed retrieval, and agentic decision-making.

### 🖥️ Frontend (Client)
* **Framework:** React 18 + Vite
* **Styling:** Vanilla CSS (Glassmorphism, custom dark themes) + Lucide Icons
* **Animations:** Framer Motion
* **Communication:** Server-Sent Events (SSE) for real-time word-by-word streaming and tool-call rendering.

### ⚙️ Backend (Server)
* **Framework:** FastAPI (Python) + Uvicorn
* **Agent Logic:** LangGraph (StateGraph) + LangChain
* **Vector Store:** Hybrid approach combining **Pinecone** (Cloud Dense Vectors) and **BM25** (Local Sparse Vectors).
* **Caching:** Semantic Query Caching to prevent redundant LLM generation on identical questions.

---

## 2. Agentic Workflow (LangGraph)

The core intelligence of the backend operates on a Directed Acyclic Graph (DAG) built with LangGraph. It features a dynamic router that directs user queries to specialized "Nodes" based on semantic intent.

1. **LLM Router Node:** Analyzes the prompt using Llama-3.1 and semantically classifies it into 5 distinct paths.
2. **📚 RAG Node:** Handles personal queries. Uses Hybrid Search (Dense + Sparse) to fetch documents, reranks them using a CrossEncoder, and strictly answers based on context.
3. **🧮 Math Node:** Extracts pure arithmetic formulas from natural language and safely evaluates them.
4. **🐍 Python Node:** Generates and safely executes Python code in a sandbox environment to solve logical or algorithmic queries.
5. **🌐 Web Node:** Uses the DuckDuckGo Search API to browse the internet in real-time for live data (e.g., stock prices, news).
6. **🧠 LLM Node:** A fallback standard chat interface for general conversational knowledge and greetings.

---

## 3. AI Models in Use

This system relies on three separate AI models, optimized for speed and cost-efficiency:

| Task | Model | Provider | Cost |
| :--- | :--- | :--- | :--- |
| **Generation & Routing** | `llama-3.1-8b-instant` | Groq API | Free Tier / Ultra-Fast |
| **Dense Embeddings** | `all-MiniLM-L6-v2` | HuggingFace (Local CPU) | Free (Open Source) |
| **Context Reranking** | `ms-marco-MiniLM-L-6-v2` | SentenceTransformers (Local CPU) | Free (Open Source) |

---

## 4. Deployment Guide

Because the application is decoupled, the Frontend and Backend should ideally be deployed to separate hosting services.

### Step 1: Environment Variables
Wherever you deploy the backend, you must configure the following secure environment variables:
```env
GROQ_API_KEY="your-groq-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="your-index-name"
```

### Step 2: Backend Deployment (Render, Railway, or Heroku)
The backend is a standard FastAPI application. **Render.com** is highly recommended for Python deployments.
1. Create a new Web Service on Render and link your GitHub repository.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add your Environment Variables in the Render dashboard.
5. Deploy! Render will provide a public URL (e.g., `https://neurag-api.onrender.com`).

### Step 3: Frontend Deployment (Vercel or Netlify)
**Vercel** is the industry standard for Vite/React applications.
1. Before deploying, update `API_URL` in `frontend/src/App.jsx` from `http://localhost:8000` to your new live backend URL (e.g., `https://neurag-api.onrender.com`).
2. Go to Vercel.com and link your GitHub repository.
3. Set the Root Directory to `frontend`.
4. **Build Command:** `npm run build`
5. **Install Command:** `npm install`
6. Deploy! Vercel will instantly build and host your frontend globally on a CDN.

---

### Step 4 (Alternative): Docker Compose Deployment
If you prefer to deploy everything on a single VPS (like AWS EC2 or DigitalOcean Droplet), you can create a `docker-compose.yml` file to spin up both containers simultaneously. Ensure you bind the Frontend to port `80` and the Backend to port `8000`, and configure your reverse proxy (Nginx/Traefik) to handle CORS properly.
