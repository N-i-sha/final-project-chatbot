# 🎓 AI-Powered College FAQ Chatbot (RAG)

**Final Year B.Tech Project**  
A full-stack Retrieval-Augmented Generation (RAG) chatbot that answers college-related questions using a local vector database and the Groq LLM API.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Backend | Python 3.10+ + FastAPI |
| RAG Framework | LangChain |
| Vector Database | ChromaDB (local, persisted to disk) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (free, local) |
| LLM | Groq API — `llama-3.1-8b-instant` (free tier) |
| API Communication | Axios |

---

## 📁 Project Structure

```
my-chatbot/
├── backend/
│   ├── app.py              ← FastAPI server (routes, CORS, chat history)
│   ├── rag_engine.py       ← Full RAG pipeline (load→embed→retrieve→answer)
│   ├── config.py           ← All settings in one place
│   ├── requirements.txt    ← Python dependencies
│   ├── .env                ← API key file (never share this!)
│   └── data/
│       └── college_faq.txt ← Knowledge base (edit this to add more FAQs)
└── frontend/
    ├── src/
    │   ├── App.jsx                     ← Root component, Axios calls
    │   ├── main.jsx                    ← React entry point
    │   ├── components/
    │   │   ├── ChatWindow.jsx          ← Scrollable message list
    │   │   ├── MessageBubble.jsx       ← Individual message (user/bot)
    │   │   └── InputBar.jsx            ← Text input + send button
    │   └── index.css                   ← Tailwind + custom animations
    ├── index.html
    ├── vite.config.js                  ← Proxy /api → localhost:8000
    ├── package.json
    ├── tailwind.config.js
    └── postcss.config.js
```

---

## ⚙️ Prerequisites

- **Python 3.10 or higher** — [python.org/downloads](https://www.python.org/downloads/)
- **Node.js 18 or higher** — [nodejs.org](https://nodejs.org/)

---

## 🔑 Step 1 — Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Click **"Create API Key"** and copy your key
4. The free tier provides ~14,400 requests/day — more than enough for a demo

---

## 🐍 Step 2 — Backend Setup

Open a terminal and run the following commands:

```powershell
# Navigate to the backend folder
cd my-chatbot\backend

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Install all dependencies one by one
pip install pydantic==1.10.21
pip install fastapi==0.100.1 uvicorn[standard]==0.23.2
pip install python-multipart==0.0.9 httpx==0.27.0 python-dotenv==1.0.1
pip install langchain==0.2.6 langchain-community==0.2.6 langchain-groq==0.1.6
pip install chromadb==0.4.24
pip install sentence-transformers==3.0.1
```

---

## 🔐 Step 3 — API Key Setup (.env file)

Create a file named `.env` inside the `backend/` folder and add the following line:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

> ⚠️ Never share your `.env` file — it contains your private API key.

The `config.py` file already has the following code which automatically reads the `.env` file every time the server starts:

```python
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
```

You do **not** need to set the API key manually in the terminal every time.

---

## ⚛️ Step 4 — Frontend Setup

Open a **new terminal** and run:

```powershell
# Navigate to the frontend folder
cd my-chatbot\frontend

# Install all Node.js packages
npm install

# If you get a tailwindcss error, also run this
npm install tailwindcss@3.4.4 postcss autoprefixer --save-dev
```

---

## 🚀 Step 5 — Run the Project

You need **two terminals open at the same time**.

### Terminal 1 — Start the Backend:
```powershell
cd my-chatbot\backend
venv\Scripts\activate
python app.py
```

✅ You will see this when the backend is ready:
```
INFO: RAG engine ready. Server is live.
INFO: Uvicorn running on http://0.0.0.0:8000
```

> **First run only:** The embedding model (~22MB) will be downloaded and ChromaDB will be built from your FAQ file. This takes 1–2 minutes. Every run after that will start instantly.

### Terminal 2 — Start the Frontend:
```powershell
cd my-chatbot\frontend
npm run dev
```

✅ You will see this when the frontend is ready:
```
VITE v5.x.x  ready in 500 ms
➜  Local:   http://localhost:5173/
```

### Open in Browser:
```
http://localhost:5173
```

---

## 💬 How It Works

```
User types a question
        ↓
React sends POST /api/chat  →  Vite proxy  →  FastAPI /chat
        ↓
FastAPI calls RAGEngine.query()
        ↓
sentence-transformers converts the question into a vector
        ↓
ChromaDB returns the top 3 most similar FAQ chunks
        ↓
Chunks + question + chat history → Groq LLM (llama-3.1-8b-instant)
        ↓
LLM generates a grounded, context-aware answer
        ↓
FastAPI returns { response, sources }
        ↓
React displays the answer in a chat bubble
```

---

## 📝 How to Add New FAQ Documents

1. Create a new `.txt` file inside `backend/data/` (e.g., `sports_faq.txt`)
2. Write Q&A pairs in the same format as `college_faq.txt`
3. Rebuild the ChromaDB index so the new content is indexed:

```powershell
# Delete the old index and restart the server
cd my-chatbot\backend
Remove-Item -Recurse -Force chroma_db
python app.py
```

The server will automatically rebuild the index from all files in the `data/` folder.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — confirms the server is running |
| `POST` | `/chat` | Send a user message, receive a bot answer |
| `POST` | `/reset` | Clear the conversation history |
| `POST` | `/rebuild-index` | Rebuild ChromaDB from the data folder |

**Interactive API documentation** is available at:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs) (when backend is running)

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `model_decommissioned` error | Change model to `llama-3.1-8b-instant` in `config.py` |
| `Failed building wheel for pydantic-core` | Use `pip install pydantic==1.10.21` instead |
| `'source' is not recognized` | On Windows use `venv\Scripts\activate` not `source venv/bin/activate` |
| `No module named 'backend'` | In `rag_engine.py` use `import config` not `import backend.config` |
| `sqlite3: no such column` error | Delete the `chroma_db` folder and restart the server |
| `Cannot find module 'tailwindcss'` | Run `npm install tailwindcss@3.4.4 postcss autoprefixer --save-dev` |
| `CORS error` in browser | Make sure the backend is running on port 8000 before opening the frontend |
| Slow first response | The embedding model downloads on the first run only — this is normal |
| `npm: command not found` | Install Node.js 18+ from [nodejs.org](https://nodejs.org/) |
| `GROQ_API_KEY is not set` | Create a `.env` file in `backend/` with your API key |

---

## 📊 Project Keywords (for Report)

RAG, Retrieval-Augmented Generation, LangChain, ChromaDB, Vector Database, Cosine Similarity, Embeddings, sentence-transformers, Groq API, FastAPI, React.js, Tailwind CSS, Vite, Axios, Natural Language Processing, Information Retrieval, Conversational AI, Chatbot, Python, JavaScript

---

## 👤 Author

NISHA SONI — Final Year MCA (CSE)  
Roll No: B2492R10700018  | Batch: 2024–2026  
Greenfield Institute of Technology

