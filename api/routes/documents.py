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
    
@router.get("/list")
async def list_documents():
    """
    List all documents in the collection
    Returns document sources with chunk counts
    """
    try:
        doc_service = get_document_service()
        documents = doc_service.list_documents()
        
        return {
            "documents": documents,
            "total_documents": sum(doc["chunk_count"] for doc in documents),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/source/{source_name}")
async def delete_document_by_source(source_name: str):
    """
    Delete all chunks from a specific document source
    """
    try:
        doc_service = get_document_service()
        deleted_count = doc_service.delete_by_source(source_name)
        
        logger.info(f"🗑️ Deleted {deleted_count} chunks from {source_name}")
        
        return {
            "status": "success",
            "source": source_name,
            "chunks_deleted": deleted_count,
            "message": f"Deleted {deleted_count} chunks from {source_name}"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/all")
async def clear_all_documents():
    """
    Clear all documents from the collection
    WARNING: This cannot be undone!
    """
    try:
        doc_service = get_document_service()
        deleted_count = doc_service.clear_all()
        
        logger.warning(f"🗑️ Cleared ALL documents ({deleted_count} chunks)")
        
        return {
            "status": "success",
            "chunks_deleted": deleted_count,
            "message": f"Cleared all {deleted_count} chunks from the collection"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))