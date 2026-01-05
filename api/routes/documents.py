from fastapi import APIRouter, HTTPException, UploadFile, File
from datetime import datetime
from api.models.responses import DocumentUploadResponse
from api.services.document_service import get_document_service
from utils.logger import logger

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    try:
        doc_service = get_document_service()
        result = await doc_service.upload_and_process(file)
        
        return DocumentUploadResponse(
            status=result["status"],
            filename=result["filename"],
            chunks_created=result["chunks_created"],
            message=f"Document '{result['filename']}' processed successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_collection_info():
    """Get information about the document collection"""
    try:
        doc_service = get_document_service()
        info = doc_service.get_collection_info()
        
        return {
            **info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))