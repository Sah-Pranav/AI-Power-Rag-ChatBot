from app.ingestion.pdf_loader import partition_document, create_chunks_by_title
from app.pipeline.document_normalizer import build_documents

pdf_file = "./docs/sample.pdf"  # replace with a small PDF for testing

elements = partition_document(pdf_file)
chunks = create_chunks_by_title(elements)

print(f"Total elements extracted: {len(elements)}")
print(f"Total chunks created: {len(chunks)}")

# # Optional: preview first chunk
# if chunks:
#     print("First chunk preview:")
#     print(chunks[0].text[:500])


documents = build_documents(
    chunks=chunks,
    source_name="sample.pdf"
)

print(f"Documents ready for embedding: {len(documents)}")

# Preview first document
if documents:
    print("\nFirst document preview:")
    print(documents[0].page_content[:500])
    print("\nMetadata:")
    print(documents[0].metadata)

