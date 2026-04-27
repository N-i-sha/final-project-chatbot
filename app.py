"""
app.py — FastAPI main server for the College FAQ Chatbot.
Exposes REST endpoints that the React frontend calls via Axios.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Single global RAG engine instance (shared across all requests) ────────────
rag_engine = RAGEngine()

# ── In-memory chat history store (keyed by session; simplified to one session) ─
# Stores last N user+assistant messages for context
chat_history: list[dict] = []


# ── Startup / Shutdown lifecycle ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine when the server starts up."""
    logger.info("Server starting — initializing RAG engine...")
    rag_engine.initialize()
    logger.info("RAG engine ready. Server is live.")
    yield
    # Nothing special needed on shutdown
    logger.info("Server shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="College FAQ Chatbot API",
    description="RAG-powered chatbot for college-related FAQs",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow requests from the React dev server (port 5173) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,  # ["http://localhost:5173", ...]
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS, etc.
    allow_headers=["*"],   # Content-Type, Authorization, etc.
)


# ── Pydantic models (request / response shapes) ───────────────────────────────
class ChatRequest(BaseModel):
    message: str           # The user's question text


class ChatResponse(BaseModel):
    response: str          # The bot's answer
    sources: list[str]     # FAQ file names used to generate the answer


class StatusResponse(BaseModel):
    status: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=StatusResponse)
def health_check():
    """
    GET /
    Simple health-check route. The frontend or monitoring tool can ping this
    to confirm the backend is running.
    """
    return {"status": "ok", "message": "College FAQ Chatbot API is running."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    POST /chat
    Accepts a user message, runs the RAG pipeline, returns the bot answer.

    Request body:  { "message": "What is the fee structure?" }
    Response body: { "response": "...", "sources": ["college_faq.txt"] }
    """
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if len(user_message) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Message too long. Please keep it under 1000 characters."
        )

    try:
        # Pass the current chat history so the LLM has conversation context
        result = rag_engine.query(
            user_question=user_message,
            chat_history=chat_history,
        )
    except RuntimeError as e:
        logger.error(f"RAG engine error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during /chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again."
        )

    # ── Append this turn to history (keep only last HISTORY_LIMIT turns) ──
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": result["answer"]})

    # Trim to last HISTORY_LIMIT Q+A pairs (each pair = 2 entries)
    max_entries = config.HISTORY_LIMIT * 2
    if len(chat_history) > max_entries:
        del chat_history[:-max_entries]

    return {"response": result["answer"], "sources": result["sources"]}


@app.post("/reset", response_model=StatusResponse)
def reset_chat():
    """
    POST /reset
    Clears the in-memory chat history so the next conversation starts fresh.
    """
    chat_history.clear()
    logger.info("Chat history cleared.")
    return {"status": "ok", "message": "Chat history has been reset."}


@app.post("/rebuild-index", response_model=StatusResponse)
def rebuild_index():
    """
    POST /rebuild-index
    Deletes and rebuilds the ChromaDB vector index from the data/ folder.
    Call this after adding new FAQ documents.
    """
    try:
        rag_engine.rebuild_index()
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

    return {"status": "ok", "message": "ChromaDB index rebuilt successfully."}


# ── Entry point (run directly with: python app.py) ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=True,   # Auto-restart when code changes (dev mode)
    )
