"""
config.py — Central configuration for the RAG FAQ Chatbot.
All tunable settings live here so you only need to edit one file.
"""


import os
from dotenv import load_dotenv

# .env file se automatically API key load karo
load_dotenv()

# ─── Groq LLM Settings ────────────────────────────────────────────────────────
# Get your FREE key at: https://console.groq.com
# Set it as an environment variable:  export GROQ_API_KEY="your_key_here"
# Or paste it directly below (not recommended for production)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")

# The model to use via Groq API (llama3-8b is fast and free)
GROQ_MODEL: str = "llama-3.1-8b-instant"

# Controls randomness of the LLM response (0 = deterministic, 1 = creative)
TEMPERATURE: float = 0.2

# Maximum number of tokens the LLM can generate in its answer
MAX_TOKENS: int = 512

# ─── Document & Chunking Settings ─────────────────────────────────────────────
# Folder where your FAQ .txt or .pdf files live
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")

# Characters per chunk when splitting the FAQ document
CHUNK_SIZE: int = 500

# Overlap between consecutive chunks (helps preserve context across boundaries)
CHUNK_OVERLAP: int = 100

# ─── Vector Store Settings ────────────────────────────────────────────────────
# ChromaDB will persist embeddings here so they aren't rebuilt on every restart
CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")

# Name of the ChromaDB collection that stores our FAQ embeddings
COLLECTION_NAME: str = "college_faq"

# ─── Retrieval Settings ───────────────────────────────────────────────────────
# How many relevant chunks to retrieve from ChromaDB per user query
TOP_K: int = 3

# ─── Chat History ─────────────────────────────────────────────────────────────
# Number of past Q&A pairs to include in each LLM prompt (for context)
HISTORY_LIMIT: int = 5

# ─── Embedding Model ──────────────────────────────────────────────────────────
# Free model from HuggingFace; runs entirely on your local machine
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─── Server ───────────────────────────────────────────────────────────────────
BACKEND_HOST: str = "0.0.0.0"
BACKEND_PORT: int = 8000

# Origins allowed to call the API (React dev server runs on 5173)
ALLOWED_ORIGINS: list = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# ─── Fallback Response ────────────────────────────────────────────────────────
# Returned when retrieved chunks are not relevant enough to answer the query
FALLBACK_RESPONSE: str = (
    "I don't have specific information about that. "
    "Please contact the college office directly or visit the official website."
)
