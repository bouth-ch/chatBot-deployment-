import os
import faiss
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from .config import AppConfig


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "If context is retrieved, use it. "
    "If not, answer concisely from general knowledge."
)


class RagChatEngine:
    """Query engine using FAISS locally + Groq LLM."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Embeddings and LLM
        Settings.embed_model = HuggingFaceEmbedding(model_name=config.embed_model)
        Settings.llm = Groq(api_key=config.groq_api_key, model=config.groq_model)

        # Load FAISS index
        if os.path.exists(self.config.faiss_index_path):
            faiss_index = faiss.read_index(self.config.faiss_index_path)
        else:
            dim = getattr(Settings.embed_model, "dimension", None) or 384
            faiss_index = faiss.IndexFlatIP(dim)

        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Load LlamaIndex storage
        if os.path.isdir(self.config.persist_dir) and os.listdir(self.config.persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_dir)
            try:
                index = load_index_from_storage(storage_context, vector_store=vector_store)
            except Exception:
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        else:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        self.retriever = index.as_retriever(similarity_top_k=config.top_k)
        self.chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            similarity_top_k=config.top_k,
            system_prompt=SYSTEM_PROMPT,
            response_mode="compact",
        )

    def chat(self, question: str) -> str:
        # Fallback if nothing retrieved
        nodes = self.retriever.retrieve(question)
        if not nodes:
            return Settings.llm.complete(question).text
        resp = self.chat_engine.chat(question)
        return str(resp)
