from datetime import datetime
from typing import Dict, Optional
from app.summarizer.ai_summary import get_rag_pipeline
from utils.logger import logger

class RAGService:
    def __init__(self):
        self.rag_pipeline = get_rag_pipeline()

    def query_documents(self, question: str, top_k: int = None, source: Optional[str] = None) -> Dict:
        start_time = datetime.utcnow()
        logger.info(f"📝 Processing query: {question} | source={source}")

        result = self.rag_pipeline.query(
            question=question,
            top_k=top_k,
            source=source,         # ✅ pass through
            return_sources=True
        )

        query_time = (datetime.utcnow() - start_time).total_seconds()
        result["query_time"] = query_time
        return result

_rag_service = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
