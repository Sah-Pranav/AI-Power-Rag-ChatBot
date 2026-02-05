# scripts/run_full_pipeline.py

"""
Complete RAG pipeline test with research paper PDF using PyMuPDF
"""

from app.ingestion.pymupdf_loader import load_and_process_pdf
from app.embeddings.vectorstore import get_vectorstore
from app.summarizer.ai_summary import get_rag_pipeline
from utils.logger import logger

def main():
    print("\n" + "="*70)
    print("🚀 COMPLETE RAG PIPELINE - RESEARCH PAPER")
    print("="*70 + "\n")
    
    # Step 1: Load and process PDF with PyMuPDF
    pdf_file = "./docs/sample.pdf"
    
    logger.info(f"📄 Step 1: Loading PDF with PyMuPDF: {pdf_file}")
    documents = load_and_process_pdf(pdf_file, source_name="app_build_paper.pdf")
    print(f"✅ Processed PDF into {len(documents)} document chunks\n")
    
    # Preview first document
    if documents:
        print("📋 First document preview:")
        print(f"   Content: {documents[0].page_content[:300]}...")
        print(f"   Metadata: {documents[0].metadata}\n")
    
    # Step 2: Add to vector store
    logger.info("💾 Step 2: Adding documents to vector store...")
    vectorstore = get_vectorstore()
    doc_ids = vectorstore.add_documents(documents)
    print(f"✅ Added {len(doc_ids)} documents to vector store\n")
    
    # Check total documents in collection
    total_docs = vectorstore.get_collection_count()
    print(f"📊 Total documents in collection: {total_docs}\n")
    
    # Step 3: Initialize RAG pipeline
    logger.info("🤖 Step 3: Initializing RAG pipeline...")
    rag = get_rag_pipeline()
    print("✅ RAG pipeline ready\n")
    
    # Step 4: Ask questions about the research paper
    print("="*70)
    print("💬 ASKING QUESTIONS ABOUT THE RESEARCH PAPER")
    print("="*70 + "\n")
    
    # Questions about your app.build research paper
    test_queries = [
        "What is app.build and what problem does it solve?",
        # "Who are the authors of this paper? List their names and affiliations.",
        # "What is environment scaffolding and how does it work?",
        # "What are the key differences between model-centric generation and environment scaffolding?",
        # "What were the main experimental results and findings?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"❓ Question {i}: {query}")
        print('─'*70)
        
        result = rag.query(query, top_k=3)
        
        print(f"\n🤖 Answer:\n{result['answer']}\n")
        
        print(f"📚 Sources ({result['retrieved_docs']} documents retrieved):")
        for j, source in enumerate(result['sources'], 1):
            print(f"  {j}. {source['source']} (Page {source['page']}) - Relevance: {source['relevance']}")
            print(f"     {source['content_preview'][:120]}...\n")
        
        print()
    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE TEST FINISHED!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()