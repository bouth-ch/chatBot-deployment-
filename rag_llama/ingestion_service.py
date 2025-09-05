# rag_llama/ingestion_service.py

from pathlib import Path
from typing import Optional, Dict, Any

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from .config import AppConfig


class IngestionService:
    def __init__(self, config: AppConfig):
        self.config = config

        # paths
        self.index_path = Path(config.index_path)
        self.storage_path = Path(config.storage_path)

        # embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=config.embedding_model)

        # load or init FAISS
        self.index = None
        if self.index_path.exists():
            # load FAISS
            faiss_index = faiss.read_index(str(self.index_path))
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path),
                vector_store=vector_store,
            )
            self.index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        else:
            # fresh index
            vector_store = FaissVectorStore.from_params(
                dim=self.embed_model.dim
            )
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path),
                vector_store=vector_store,
            )
            self.index = VectorStoreIndex.from_documents(
                [], storage_context=storage_context, embed_model=self.embed_model
            )

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Insert plain text into the index."""
        doc = Document(text=text, metadata=metadata or {})
        self.index.insert(doc)
        self.persist()
        return {"chunks_indexed": 1}

    def ingest_txt_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Handle .txt file upload."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("File must be UTF-8 encoded text")

        stats = self.ingest_text(text, metadata={"filename": filename})
        stats["filename"] = filename
        return stats

    def persist(self):
        """Save FAISS index and storage context."""
        faiss.write_index(self.index.vector_store._faiss_index, str(self.index_path))
        self.index.storage_context.persist(persist_dir=str(self.storage_path))
