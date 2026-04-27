"""
rag_engine.py — The core RAG (Retrieval-Augmented Generation) pipeline.
Loads FAQ documents, creates embeddings, stores them in ChromaDB,
and answers user queries using retrieved context + Groq LLM.
"""

import os
import glob
import logging
from typing import Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import Document

import config

# Set up logging so we can see what's happening at runtime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    End-to-end RAG engine for the College FAQ chatbot.
    
    Flow:
      1. Load .txt/.pdf documents from data/ folder
      2. Split them into overlapping chunks
      3. Generate embeddings using sentence-transformers (local, free)
      4. Store embeddings in ChromaDB (persisted to disk)
      5. On each query: retrieve top-K relevant chunks, build a prompt,
         send it to Groq LLM, and return the answer
    """

    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.llm: Optional[ChatGroq] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialized: bool = False

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Fully set up the RAG pipeline.
        Called once when the FastAPI server starts.
        """
        logger.info("Initializing RAG engine...")

        # Step 1: Load the embedding model (downloads once, cached locally)
        self._load_embeddings()

        # Step 2: Load or build the ChromaDB vector store
        self._initialize_vectorstore()

        # Step 3: Connect to the Groq LLM
        self._initialize_llm()

        self._initialized = True
        logger.info("RAG engine ready.")

    def _load_embeddings(self) -> None:
        """
        Load the sentence-transformers embedding model.
        'all-MiniLM-L6-v2' is small (22MB), fast, and produces good results.
        """
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            # Cache downloaded model in the project folder
            cache_folder=os.path.join(os.path.dirname(__file__), ".model_cache"),
            model_kwargs={"device": "cpu"},   # Use CPU (no GPU required)
            encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
        )
        logger.info("Embedding model loaded.")

    def _load_documents(self) -> list[Document]:
        """
        Read all .txt files from the data/ directory and return as LangChain Documents.
        Each Document has 'page_content' (text) and 'metadata' (source file path).
        """
        if not os.path.exists(config.DATA_DIR):
            raise FileNotFoundError(f"Data directory not found: {config.DATA_DIR}")

        # Use DirectoryLoader to load all .txt files at once
        loader = DirectoryLoader(
            config.DATA_DIR,
            glob="**/*.txt",         # Match all .txt files recursively
            loader_cls=TextLoader,   # Use simple text loader
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )

        documents = loader.load()

        if not documents:
            raise ValueError(f"No .txt documents found in {config.DATA_DIR}")

        logger.info(f"Loaded {len(documents)} document(s) from {config.DATA_DIR}")
        return documents

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split large documents into smaller overlapping chunks.
        Smaller chunks = more precise retrieval.
        Overlap ensures sentences at boundaries aren't cut off.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,        # Target characters per chunk
            chunk_overlap=config.CHUNK_OVERLAP,  # Characters shared between adjacent chunks
            length_function=len,
            # Try to split at paragraph → newline → sentence → word boundaries
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")
        return chunks

    def _initialize_vectorstore(self) -> None:
        """
        Load ChromaDB from disk if it already exists (fast startup),
        otherwise build it from scratch by embedding the FAQ documents.
        """
        # Check if a persisted ChromaDB already exists
        chroma_exists = (
            os.path.exists(config.CHROMA_PERSIST_DIR) and
            len(os.listdir(config.CHROMA_PERSIST_DIR)) > 0
        )

        if chroma_exists:
            logger.info("Loading existing ChromaDB from disk...")
            # Simply load the previously persisted database — no re-embedding
            self.vectorstore = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR,
            )
            count = self.vectorstore._collection.count()
            logger.info(f"ChromaDB loaded with {count} vectors.")
        else:
            logger.info("No existing ChromaDB found. Building from documents...")
            # Load → split → embed → store (this only happens on the first run)
            documents = self._load_documents()
            chunks = self._split_documents(documents)

            # Chroma.from_documents() does all embedding and storage in one call
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.CHROMA_PERSIST_DIR,
            )
            logger.info(f"ChromaDB built and saved with {len(chunks)} chunks.")

    def _initialize_llm(self) -> None:
        """
        Connect to the Groq API and set up the ChatGroq LLM instance.
        Groq provides blazing-fast inference for Llama3 models (free tier available).
        """
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set! "
                "Get your free key at https://console.groq.com and "
                "set it as: export GROQ_API_KEY='your_key'"
            )

        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        logger.info(f"Groq LLM initialized with model: {config.GROQ_MODEL}")

    # ── Core Query Function ───────────────────────────────────────────────────

    def query(self, user_question: str, chat_history: list[dict]) -> dict:
        """
        Main RAG query function. Called for every user message.
        
        Args:
            user_question: The question text from the user.
            chat_history:  List of previous {"role": "...", "content": "..."} dicts.

        Returns:
            dict with keys:
              - "answer": str  (the bot's answer)
              - "sources": list[str]  (source file names used)
        """
        if not self._initialized:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        # ── Step 1: Retrieve relevant chunks from ChromaDB ──
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",            # Cosine similarity search
            search_kwargs={"k": config.TOP_K},  # Fetch top-K most relevant chunks
        )
        relevant_docs = retriever.invoke(user_question)

        if not relevant_docs:
            return {"answer": config.FALLBACK_RESPONSE, "sources": []}

        # ── Step 2: Build the context string from retrieved chunks ──
        context_parts = []
        sources = set()

        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[Chunk {i}]\n{doc.page_content}")
            # Extract just the filename from the full path
            source_path = doc.metadata.get("source", "unknown")
            sources.add(os.path.basename(source_path))

        context_text = "\n\n".join(context_parts)

        # ── Step 3: Format chat history for the prompt ──
        # Include only the last N turns so the prompt doesn't grow too large
        recent_history = chat_history[-(config.HISTORY_LIMIT * 2):]
        history_text = ""
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role = "Student" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {msg['content']}")
            history_text = "\n".join(history_lines)

        # ── Step 4: Build the full prompt ──
        prompt = self._build_prompt(
            context=context_text,
            history=history_text,
            question=user_question,
        )

        # ── Step 5: Send to Groq LLM and get the answer ──
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            answer = (
                "I encountered an error while generating the response. "
                "Please try again in a moment."
            )

        return {"answer": answer, "sources": sorted(sources)}

    def _build_prompt(self, context: str, history: str, question: str) -> str:
        """
        Construct a well-structured prompt for the LLM.
        
        The prompt includes:
        - A system persona (college assistant)
        - Relevant FAQ chunks as context
        - Recent conversation history
        - The current user question
        - Clear instructions on how to respond
        """
        history_section = ""
        if history:
            history_section = f"""
## Conversation History
{history}
"""

        prompt = f"""You are a helpful and friendly FAQ assistant for Greenfield Institute of Technology (GIT).
Your job is to answer student questions accurately based ONLY on the provided college FAQ information.

## Rules:
- Answer ONLY from the context below. Do not use external knowledge.
- If the answer is not in the context, respond with: "I don't have specific information about that. Please contact the college office or visit git.ac.in for more details."
- Be concise, friendly, and professional.
- If the question is a greeting or small talk, respond naturally without using the context.
- Format numbers, fees, and lists clearly when present in the context.
{history_section}
## Relevant FAQ Information:
{context}

## Student Question:
{question}

## Your Answer:"""

        return prompt

    # ── Utility ───────────────────────────────────────────────────────────────

    def rebuild_index(self) -> None:
        """
        Force-rebuild the ChromaDB index from documents.
        Useful when you add new FAQ documents.
        """
        import shutil
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            shutil.rmtree(config.CHROMA_PERSIST_DIR)
            logger.info("Existing ChromaDB deleted.")

        self._initialized = False
        self._initialize_vectorstore()
        self._initialized = True
        logger.info("ChromaDB rebuilt successfully.")
