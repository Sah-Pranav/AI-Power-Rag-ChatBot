from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.models.responses import HealthResponse
from api.services.document_service import get_document_service
from utils.logger import logger

router = APIRouter(tags=["Health"])

@router.get("/", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        doc_service = get_document_service()
        info = doc_service.get_collection_info()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            vector_store_status="connected",
            document_count=info["total_documents"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")