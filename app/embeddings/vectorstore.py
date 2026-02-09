# app/embeddings/vectorstore.py

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.embeddings.embedding_factory import get_embeddings
from utils.config_loader import config
from utils.logger import logger
import os


class VectorStoreManager:
    """Manage vector store operations"""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize ChromaDB vector store"""
        provider = config.get("vectorstore", "provider")

        if provider != "chroma":
            raise ValueError(f"❌ Unsupported vector store provider: {provider}")

        persist_dir = config.get("vectorstore", "chroma", "persist_directory")
        collection_name = config.get("vectorstore", "chroma", "collection_name")

        os.makedirs(persist_dir, exist_ok=True)

        logger.info(f"🗄️  Initializing ChromaDB at: {persist_dir}")
        logger.info(f"📚 Collection name: {collection_name}")

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        logger.info("✅ Vector store initialized successfully")

    def _maybe_reinit_on_missing_collection(self, e: Exception) -> bool:
        msg = str(e)
        if "does not exist" in msg and "Collection" in msg:
            logger.warning("🔄 Collection missing. Reinitializing vectorstore and retrying...")
            self._initialize_vectorstore()
            return True
        return False

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            logger.warning("⚠️ No documents to add")
            return []

        logger.info(f"📥 Adding {len(documents)} documents to vector store...")

        try:
            return self.vectorstore.add_documents(documents)

        except Exception as e:
            if self._maybe_reinit_on_missing_collection(e):
                return self.vectorstore.add_documents(documents)
            logger.error(f"❌ Failed to add documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        if k is None:
            k = config.get("retrieval", "top_k", default=5)

        logger.info(f"🔍 Searching for: '{query}' (top {k} results)")

        try:
            return self.vectorstore.similarity_search(query, k=k)

        except Exception as e:
            if self._maybe_reinit_on_missing_collection(e):
                return self.vectorstore.similarity_search(query, k=k)
            logger.error(f"❌ Search failed: {e}")
            raise

    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        if k is None:
            k = config.get("retrieval", "top_k", default=5)

        logger.info(f"🔍 Searching with scores: '{query}' (top {k})")

        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)

        except Exception as e:
            if self._maybe_reinit_on_missing_collection(e):
                return self.vectorstore.similarity_search_with_score(query, k=k)
            logger.error(f"❌ Search with scores failed: {e}")
            raise

    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5) -> List[Document]:
        """
        Max Marginal Relevance search for diversity.
        """
        logger.info(f"🔍 MMR search: query='{query}' k={k} fetch_k={fetch_k} lambda={lambda_mult}")

        try:
            return self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )

        except Exception as e:
            if self._maybe_reinit_on_missing_collection(e):
                return self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )
            logger.error(f"❌ MMR search failed: {e}")
            raise

    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        try:
            return int(self.vectorstore._collection.count())

        except Exception as e:
            if self._maybe_reinit_on_missing_collection(e):
                return int(self.vectorstore._collection.count())
            logger.error(f"❌ Failed to get collection count: {e}")
            return 0


_vectorstore_instance = None


def get_vectorstore() -> VectorStoreManager:
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStoreManager()
    return _vectorstore_instance


def reset_vectorstore():
    """Reset the singleton vectorstore (forces re-init on next get_vectorstore call)."""
    global _vectorstore_instance
    _vectorstore_instance = None
