from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(None, ge=1, le=10)

    # ✅ NEW: optional source filter (PDF filename)
    source: Optional[str] = Field(None, min_length=1, max_length=300)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is environment scaffolding?",
                "top_k": 3,
                "source": "app_buid_paper.pdf"
            }
        }
