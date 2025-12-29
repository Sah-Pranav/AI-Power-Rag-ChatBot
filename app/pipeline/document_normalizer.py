# app/pipeline/document_normalizer.py

import re
from typing import List
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean OCR / PDF extracted text.
    """
    if not text:
        return ""

    # Remove excessive newlines
    text = re.sub(r"\n+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", "", text)

    return text.strip()


def build_documents(
    chunks,
    source_name: str,
) -> List[Document]:
    """
    Convert Unstructured chunks into LangChain Documents.
    """
    documents: List[Document] = []

    for chunk in chunks:
        raw_text = getattr(chunk, "text", "")
        cleaned_text = clean_text(raw_text)

        # Drop garbage / very small chunks
        if len(cleaned_text) < 80:
            continue

        metadata = {
            "source": source_name,
            # ElementMetadata → attribute access (NOT dict)
            "page": getattr(chunk.metadata, "page_number", None),
            "category": getattr(chunk, "category", None),
        }

        documents.append(
            Document(
                page_content=cleaned_text,
                metadata=metadata,
            )
        )

    return documents