from typing import Dict, Optional
import re
from langchain_core.prompts import ChatPromptTemplate
from app.summarizer.llm_factory import get_llm
from app.retriever.query import get_retriever
from utils.logger import logger

def _strip_leading_junk(text: str) -> str:
    t = (text or "").lstrip()
    while t.startswith("...") or t.startswith("…") or t.startswith("."):
        t = t.lstrip(".… ").lstrip()
    return t

def _preview_text(text: str, max_chars: int = 450) -> str:
    raw = _strip_leading_junk(text)

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)  # join line-wrap newlines
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    raw = raw.strip()

    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars].rstrip() + "…"

class RAGPipeline:
    def __init__(self):
        self.llm = get_llm()
        self.retriever = get_retriever()
        self._setup_prompt()

    def _setup_prompt(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical assistant answering questions using provided documents.

RULES:
1. Use ONLY the provided documents.
2. If a definition exists, quote or paraphrase the definition clearly.
3. Prefer definition sentences over general context.
4. Do not say information is missing if relevant sentences are present.
5. Be concise and factual.
6. ALWAYS answer in English only.

Context:
{context}
"""),
            ("human", "{question}")
        ])

    def query(self, question: str, top_k: int = None, source: Optional[str] = None, return_sources: bool = True) -> Dict:
        logger.info(f"❓ Processing: '{question}' | source={source}")

        retrieved = self.retriever.retrieve(
            question,
            top_k=top_k,
            with_scores=True,
            source=source,
        )

        if not retrieved:
            return {"answer": "I could not find relevant information to answer your question.", "sources": [], "retrieved_docs": 0}

        context = self.retriever.format_context(retrieved)

        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        result: Dict = {"answer": response.content, "retrieved_docs": len(retrieved)}

        if return_sources:
            sources = []
            for doc, score in retrieved:
                rel: Optional[float] = None
                try:
                    if score is not None and float(score) > 0:
                        rel = round(1 / (1 + float(score)), 3)
                except Exception:
                    rel = None

                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "relevance": rel,
                    "content_preview": _preview_text(doc.page_content, max_chars=450),
                })
            result["sources"] = sources

        return result

def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()
