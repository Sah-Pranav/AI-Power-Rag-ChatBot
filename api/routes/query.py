from fastapi import APIRouter, HTTPException
from api.models.requests import QueryRequest
from api.models.responses import QueryResponse
from api.services.rag_service import get_rag_service
from utils.logger import logger

router = APIRouter(prefix="/query", tags=["Query"])

@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system
    
    - **question**: Your question (3-500 characters)
    - **top_k**: Number of documents to retrieve (optional)
    """
    try:
        rag_service = get_rag_service()
        result = rag_service.query_documents(
            question=request.question,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            retrieved_docs=result.get("retrieved_docs", 0),
            query_time=result["query_time"]
        )
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))