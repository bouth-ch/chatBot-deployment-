import os
from pathlib import Path

import faiss
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from .config import AppConfig


class IndexBuilder:
    """Builds a FAISS-backed index from local documents and persists it to disk."""

    def __init__(self, config: AppConfig):
        self.config = config
        Path(self.config.persist_dir).mkdir(parents=True, exist_ok=True)

        # Configure embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.config.embed_model)

    def load_documents(self):
        reader = SimpleDirectoryReader(input_dir=self.config.data_dir, recursive=True)
        return reader.load_data()

    def _build_faiss_index(self, dim: int):
        # Using Inner Product (IP) with normalized vectors ~ cosine similarity
        index = faiss.IndexFlatIP(dim)
        return index

    def build_and_persist(self):
        docs = self.load_documents()
        if not docs:
            raise RuntimeError(f"No documents found in: {self.config.data_dir}")

        # Determine embedding dimension (MiniLM -> 384)
        dim = getattr(Settings.embed_model, "dimension", None) or 384

        # Create a fresh FAISS index
        faiss_index = self._build_faiss_index(dim)

        # Wrap with LlamaIndex FaissVectorStore
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Build index
        index = VectorStoreIndex.from_documents(
            docs,
            vector_store=vector_store,
            show_progress=True,
        )

        # Persist LlamaIndex storage (docstore, index metadata)
        storage_context: StorageContext = index.storage_context
        storage_context.persist(persist_dir=self.config.persist_dir)

        # Persist FAISS index to disk
        faiss.write_index(faiss_index, self.config.faiss_index_path)

        print(
            f"Stored {len(docs)} docs. "
            f"Persisted LlamaIndex store to: {self.config.persist_dir} "
            f"and FAISS index to: {self.config.faiss_index_path}"
        )
        return index
