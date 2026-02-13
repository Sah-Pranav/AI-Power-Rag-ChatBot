from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.embeddings.embedding_factory import get_embeddings
from utils.config_loader import config
from utils.logger import logger
import os

class VectorStoreManager:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
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

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            logger.warning("⚠️ No documents to add")
            return []

        logger.info(f"📥 Adding {len(documents)} documents to vector store...")
        try:
            return self.vectorstore.add_documents(documents)
        except Exception as e:
            msg = str(e)
            if "does not exist" in msg and "Collection" in msg:
                logger.warning("🔄 Collection missing. Reinitializing vectorstore and retrying...")
                self._initialize_vectorstore()
                return self.vectorstore.add_documents(documents)
            logger.error(f"❌ Failed to add documents: {e}")
            raise

    def similarity_search_with_score(self, query: str, k: int = 5, where: Optional[Dict] = None):
        logger.info(f"🔍 similarity_search_with_score query='{query}' k={k} where={where}")
        try:
            # ✅ langchain-chroma uses `filter=` not `where=`
            return self.vectorstore.similarity_search_with_score(query, k=k, filter=where)
        except Exception as e:
            logger.error(f"❌ Search with scores failed: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5, where: Optional[Dict] = None):
        logger.info(f"🔍 similarity_search query='{query}' k={k} where={where}")
        try:
            return self.vectorstore.similarity_search(query, k=k, filter=where)
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise

    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5, where: Optional[Dict] = None):
        logger.info(f"🔍 MMR search query='{query}' k={k} fetch_k={fetch_k} lambda={lambda_mult} where={where}")
        try:
            return self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=where,   # ✅ apply source filter
            )
        except Exception as e:
            logger.error(f"❌ MMR search failed: {e}")
            raise

    def get_collection_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception as e:
            logger.error(f"❌ Failed to get collection count: {e}")
            return 0

_vectorstore_instance = None

def get_vectorstore() -> VectorStoreManager:
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStoreManager()
    return _vectorstore_instance
