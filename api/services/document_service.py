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
    
    def list_documents(self) -> list:
        """
        List all documents grouped by source
        Returns list of documents with chunk counts
        """
        try:
            # Get all documents from collection
            collection = self.vectorstore.vectorstore._collection
            
            # Get all items
            results = collection.get(include=['metadatas'])
            
            if not results or not results.get('metadatas'):
                return []
            
            # Group by source
            from collections import defaultdict
            sources = defaultdict(int)
            
            for metadata in results['metadatas']:
                source = metadata.get('source', 'Unknown')
                sources[source] += 1
            
            # Format response
            documents = [
                {
                    "name": source,
                    "chunk_count": count,
                    "source": source
                }
                for source, count in sources.items()
            ]
            
            return sorted(documents, key=lambda x: x['name'])
            
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []
    
    def delete_by_source(self, source_name: str) -> int:
        """
        Delete all chunks from a specific source
        Returns number of chunks deleted
        """
        try:
            collection = self.vectorstore.vectorstore._collection
            
            # Get all IDs for this source
            results = collection.get(
                where={"source": source_name},
                include=['metadatas']
            )
            
            if not results or not results.get('ids'):
                logger.warning(f"No documents found for source: {source_name}")
                return 0
            
            ids_to_delete = results['ids']
            
            # Delete them
            collection.delete(ids=ids_to_delete)
            
            logger.info(f"✅ Deleted {len(ids_to_delete)} chunks from {source_name}")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"❌ Failed to delete by source: {e}")
            raise
    
    def clear_all(self) -> int:
        """
        Clear all documents WITHOUT deleting the collection.
        Keeps the in-memory Chroma handle valid.
        """
        try:
            collection = self.vectorstore.vectorstore._collection

            results = collection.get(include=[])
            ids = results.get("ids", []) if results else []

            if not ids:
                logger.warning("🗑️ Collection already empty")
                return 0

            collection.delete(ids=ids)

            logger.warning(f"🗑️ Cleared all documents ({len(ids)} chunks)")
            return len(ids)

        except Exception as e:
            logger.error(f"❌ Failed to clear all: {e}")
            raise

# Singleton instance
_document_service = None

def get_document_service() -> DocumentService:
    """Get document service instance"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service