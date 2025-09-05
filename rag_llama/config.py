import os
from dotenv import load_dotenv


class AppConfig:
    """Config for Groq + FAISS local RAG."""

    def __init__(self, env_path: str = None):
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # Groq
        self.groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Put it in a .env file (see .env.example).")
        self.groq_model = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b").strip()

        # Embeddings
        self.embed_model = (os.getenv("EMBED_MODEL") or "sentence-transformers/paraphrase-MiniLM-L3-v2").strip()

        # Data / RAG
        self.data_dir = (os.getenv("DATA_DIR") or "sample_docs").strip()
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))

        # Local persistence for FAISS + LlamaIndex storage
        self.persist_dir = (os.getenv("PERSIST_DIR") or ".faiss_store").strip()
        self.faiss_index_path = os.path.join(self.persist_dir, "faiss.index")

    def __repr__(self):
        return (
            f"AppConfig(groq_model={self.groq_model}, embed_model={self.embed_model}, "
            f"data_dir={self.data_dir}, persist_dir={self.persist_dir})"
        )
