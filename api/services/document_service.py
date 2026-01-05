import os
from typing import List
from fastapi import UploadFile
from app.ingestion.pymupdf_loader import load_and_process_pdf
from app.embeddings.vectorstore import get_vectorstore
from utils.logger import logger

class DocumentService:
    """Business logic for document operations"""
    
    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.upload_dir = "./data/uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def upload_and_process(
        self, 
        file: UploadFile
    ) -> dict:
        """Upload and process a PDF document"""
        
        if not file.filename.endswith('.pdf'):
            raise ValueError("Only PDF files are supported")
        
        logger.info(f"📤 Processing: {file.filename}")
        
        temp_path = os.path.join(self.upload_dir, file.filename)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        try:
            documents = load_and_process_pdf(temp_path, source_name=file.filename)
            doc_ids = self.vectorstore.add_documents(documents)
            
            logger.info(f"✅ Processed: {len(doc_ids)} chunks")
            
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_created": len(doc_ids)
            }
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def get_collection_info(self) -> dict:
        """Get information about document collection"""
        count = self.vectorstore.get_collection_count()
        
        return {
            "total_documents": count,
            "collection_name": "document_collection"
        }

# Singleton instance
_document_service = None

def get_document_service() -> DocumentService:
    """Get document service instance"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service