# app/ingestion/pymupdf_loader.py

import re
import fitz  # PyMuPDF
from typing import List, Dict
from langchain_core.documents import Document
from utils.logger import logger


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF using PyMuPDF.
    Returns list of dicts with page_content, page_number, total_pages.
    """
    logger.info(f"📄 Extracting text from: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        pages_content = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if text and text.strip():
                pages_content.append({
                    "page_content": text,
                    "page_number": page_num + 1,
                    "total_pages": len(doc)
                })

        logger.info(f"✅ Extracted text from {len(pages_content)} pages")
        return pages_content

    except Exception as e:
        logger.error(f"❌ Failed to extract text: {e}")
        raise
    
    finally:
        doc.close()


def clean_text(text: str) -> str:
    """
    Light normalization for PDFs:
    - keep paragraph breaks
    - fix hyphenation
    - merge line-wrap newlines inside paragraphs
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenation across line breaks: "para-\ngraph" -> "paragraph"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Merge single newlines inside a paragraph into spaces (keep blank lines as paragraph breaks)
    # Example: "vs.\nmodel" -> "vs. model"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize multiple blank lines to exactly one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence split (no extra deps).
    """
    t = text.strip()
    if not t:
        return []
    sents = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in sents if s.strip()]


def strip_leading_dots(text: str) -> str:
    t = (text or "").lstrip()
    while t.startswith("...") or t.startswith("…") or t.startswith("."):
        t = t.lstrip(".… ").lstrip()
    return t


def apply_overlap(prev_chunk: str, overlap_chars: int) -> List[str]:
    """
    Create starting content for next chunk from end of previous chunk,
    trying to start at a sentence boundary.
    """
    if overlap_chars <= 0 or not prev_chunk:
        return []

    tail = prev_chunk[-overlap_chars:].strip()

    # Try to start overlap after the last sentence-ending punctuation
    m = re.search(r"([.!?])\s+[^.!?]*$", tail)
    if m:
        tail = tail[m.start(1) + 1:].strip()

    return [tail] if tail else []


def pack_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Pack paragraphs/sentences into chunks up to chunk_size with overlap.
    """
    chunks: List[str] = []
    current: List[str] = []

    def current_len() -> int:
        # length including separators
        return sum(len(x) for x in current) + max(0, len(current) - 1)

    paragraphs = split_into_paragraphs(text)

    for p in paragraphs:
        if len(p) <= chunk_size:
            if current and current_len() + 2 + len(p) > chunk_size:
                chunk = "\n\n".join(current).strip()
                chunks.append(chunk)
                current = apply_overlap(chunk, chunk_overlap)
            current.append(p)
            continue

        # paragraph too long -> sentence pack
        for s in split_into_sentences(p):
            if current and current_len() + 1 + len(s) > chunk_size:
                chunk = "\n\n".join(current).strip()
                chunks.append(chunk)
                current = apply_overlap(chunk, chunk_overlap)
            current.append(s)

    if current:
        chunks.append("\n\n".join(current).strip())

    return [strip_leading_dots(c) for c in chunks if c.strip()]


def chunk_text_by_pages(
    pages_content: List[Dict],
    source_name: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 250
) -> List[Document]:
    """
    Production-style chunking:
    - clean text (fix line-wrap)
    - preserve paragraphs
    - split long paragraphs into sentences
    - pack into chunks with overlap
    """
    logger.info("🔨 Creating chunks from pages (paragraph/sentence-aware)...")
    documents: List[Document] = []

    for page_data in pages_content:
        page_num = page_data["page_number"]
        cleaned = clean_text(page_data["page_content"])

        if len(cleaned) < 80:
            continue

        chunks = pack_text_into_chunks(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for i, chunk in enumerate(chunks):
            if len(chunk) < 120:
                continue

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
            )

    logger.info(f"✅ Created {len(documents)} document chunks")
    return documents


def load_and_process_pdf(pdf_path: str, source_name: str = None) -> List[Document]:
    import os
    if source_name is None:
        source_name = os.path.basename(pdf_path)

    pages_content = extract_text_from_pdf(pdf_path)
    return chunk_text_by_pages(pages_content, source_name)
