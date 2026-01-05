from pydantic import BaseModel
from typing import List, Dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    retrieved_docs: int
    query_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    vector_store_status: str
    document_count: int

class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    message: str