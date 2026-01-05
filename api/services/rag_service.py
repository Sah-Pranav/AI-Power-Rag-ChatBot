from datetime import datetime
from typing import Dict
from app.summarizer.ai_summary import get_rag_pipeline
from utils.logger import logger

class RAGService:
    """Business logic for RAG operations"""
    
    def __init__(self):
        self.rag_pipeline = get_rag_pipeline()
    
    def query_documents(
        self, 
        question: str, 
        top_k: int = None
    ) -> Dict:
        """Process a query and return results"""
        
        start_time = datetime.utcnow()
        logger.info(f"📝 Processing query: {question}")
        
        result = self.rag_pipeline.query(
            question=question,
            top_k=top_k,
            return_sources=True
        )
        
        end_time = datetime.utcnow()
        query_time = (end_time - start_time).total_seconds()
        
        result["query_time"] = query_time
        return result

# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service