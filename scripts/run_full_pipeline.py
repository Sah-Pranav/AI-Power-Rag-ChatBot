"""
Complete RAG pipeline smoke test with a PDF using PyMuPDF
Usage:
  python scripts/run_full_pipeline.py ./docs/app_buid_paper.pdf
"""

import os
import sys
from app.ingestion.pymupdf_loader import load_and_process_pdf
from app.embeddings.vectorstore import get_vectorstore
from app.summarizer.ai_summary import get_rag_pipeline
from utils.logger import logger


def _delete_by_source(vectorstore, source_name: str) -> int:
    """
    Best-effort delete chunks by source from Chroma collection.
    This keeps runs idempotent (no duplicates).
    """
    try:
        collection = vectorstore.vectorstore._collection
        results = collection.get(where={"source": source_name}, include=[])
        ids = results.get("ids", []) if results else []
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception as e:
        logger.warning(f"⚠️ Could not delete existing chunks for source={source_name}: {e}")
        return 0


def main():
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "./docs/sample.pdf"

    print("\n" + "=" * 70)
    print("🚀 COMPLETE RAG PIPELINE - SMOKE TEST (PyMuPDF)")
    print("=" * 70 + "\n")

    if not os.path.exists(pdf_file):
        print(f"❌ PDF not found: {pdf_file}")
        print("👉 Usage: python scripts/run_full_pipeline.py /path/to/file.pdf")
        sys.exit(1)

    source_name = os.path.basename(pdf_file)

    # Step 0: Init vectorstore
    vectorstore = get_vectorstore()

    # Step 1: Load and chunk PDF
    logger.info(f"📄 Step 1: Loading PDF with PyMuPDF: {pdf_file}")
    documents = load_and_process_pdf(pdf_file, source_name=source_name)
    print(f"✅ Processed PDF into {len(documents)} chunks\n")

    if not documents:
        print("❌ No chunks created. Extraction/chunking failed.")
        sys.exit(2)

    print("📋 First chunk preview:")
    print(f"   Content: {documents[0].page_content[:300]}...")
    print(f"   Metadata: {documents[0].metadata}\n")

    # Step 2: Remove old chunks for this source (idempotent runs)
    deleted = _delete_by_source(vectorstore, source_name)
    if deleted:
        print(f"🧹 Removed {deleted} existing chunks for source: {source_name}\n")

    # Step 3: Add to vector store
    logger.info("💾 Step 3: Adding documents to vector store...")
    doc_ids = vectorstore.add_documents(documents)
    print(f"✅ Added {len(doc_ids)} chunks to vector store\n")

    total_docs = vectorstore.get_collection_count()
    print(f"📊 Total chunks in collection now: {total_docs}\n")

    # Step 4: Query
    logger.info("🤖 Step 4: Initializing RAG pipeline...")
    rag = get_rag_pipeline()
    print("✅ RAG pipeline ready\n")

    test_queries = [
        "What is environment scaffolding?",
        "What is app.build and what problem does it solve?",
    ]

    print("=" * 70)
    print("💬 ASKING QUESTIONS")
    print("=" * 70 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"❓ Question {i}: {query}")
        print("─" * 70)

        result = rag.query(query, top_k=3)

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        retrieved = result.get("retrieved_docs", 0)

        print(f"\n🤖 Answer:\n{answer}\n")
        print(f"📚 Sources ({retrieved} retrieved):")

        for j, src in enumerate(sources, 1):
            print(f"  {j}. {src.get('source')} (Page {src.get('page')})  Relevance: {src.get('relevance')}")
            print(f"     {src.get('content_preview', '')[:160]}...\n")

        # Basic smoke assertions
        if not answer.strip():
            print("❌ Smoke check failed: empty answer")
            sys.exit(3)

    print("\n" + "=" * 70)
    print("✅ SMOKE TEST FINISHED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
