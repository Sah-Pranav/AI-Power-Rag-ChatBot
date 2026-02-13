# tests/test_smoke.py
"""
Minimal smoke test:
- Ingest a PDF (by path)
- Query one question
- Assert basic invariants
Usage:
  python -m tests.test_smoke ./docs/app_buid_paper.pdf
"""

import os
import sys

from app.ingestion.pymupdf_loader import load_and_process_pdf
from app.embeddings.vectorstore import get_vectorstore
from app.summarizer.ai_summary import get_rag_pipeline


def _delete_by_source(vectorstore, source_name: str) -> int:
    """Best-effort cleanup by source for idempotent test runs."""
    collection = vectorstore.vectorstore._collection
    results = collection.get(where={"source": source_name}, include=[])
    ids = results.get("ids", []) if results else []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./docs/sample.pdf"
    assert os.path.exists(pdf_path), f"PDF not found: {pdf_path}"

    source_name = os.path.basename(pdf_path)

    vectorstore = get_vectorstore()

    # cleanup (idempotent)
    _delete_by_source(vectorstore, source_name)

    # ingest
    docs = load_and_process_pdf(pdf_path, source_name=source_name)
    assert len(docs) > 0, "No chunks created from PDF"

    ids = vectorstore.add_documents(docs)
    assert len(ids) > 0, "No docs added to vectorstore"

    # query
    rag = get_rag_pipeline()
    out = rag.query("What is environment scaffolding?", top_k=3)

    assert out.get("answer"), "Empty answer"
    assert out.get("retrieved_docs", 0) > 0, "No retrieved docs"
    assert len(out.get("sources", [])) > 0, "No sources returned"

    print("✅ Smoke test passed.")


if __name__ == "__main__":
    main()
