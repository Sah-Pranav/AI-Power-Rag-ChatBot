# AI Power RAG ChatBot

AI Power RAG ChatBot is a tool for extracting, parsing, and reasoning over documents using a retrieval-augmented generation (RAG) pipeline. It supports PDF document processing, chunking, and integration with LLMs for advanced question answering.

---

## Features

- Parse and chunk PDFs and other document types.
- Extract meaningful elements and organize by titles.
- Supports OCR and text extraction for scanned documents.
- Connects with LLMs for Q&A using embeddings.
- Easy integration with FastAPI and Streamlit.

---

## Requirements

- Python >= 3.12
- Poppler (`pdftotext`) for PDF parsing
- Tesseract for OCR (optional)
- libmagic for file type detection

---

## Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/AIPower-Rag-ChatBot.git
cd AIPower-Rag-ChatBot

# Running the app
## Start backend
python -m api.main

## Start frontend
streamlit run frontend/app.py

