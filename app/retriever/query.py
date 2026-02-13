from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from app.embeddings.vectorstore import get_vectorstore
from utils.config_loader import config
from utils.logger import logger

class Retriever:
    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.top_k = int(config.get("retrieval", "top_k", default=5))
        self.strategy = config.get("retrieval", "strategy", default="mmr")
        self.fetch_k = int(config.get("retrieval", "fetch_k", default=max(20, self.top_k * 5)))
        self.mmr_lambda = float(config.get("retrieval", "mmr_lambda", default=0.6))

    def _strip_leading_junk(self, text: str) -> str:
        t = (text or "").lstrip()
        while t.startswith("...") or t.startswith("…") or t.startswith("."):
            t = t.lstrip(".… ").lstrip()
        return t

    def _dedupe(self, docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        seen = set()
        out: List[Tuple[Document, float]] = []
        for doc, score in docs_with_scores:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            out.append((doc, score))
        return out

    def _is_low_value_chunk(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) < 200:
            return True

        head = t[:400].lower()

        if "references" in head or "bibliography" in head:
            return True

        comma_ratio = t.count(",") / max(1, len(t))
        if comma_ratio > 0.02 and ("openai" in head or "university" in head or "google" in head or "microsoft" in head):
            return True

        digits = sum(ch.isdigit() for ch in t)
        digit_ratio = digits / max(1, len(t))
        if digit_ratio > 0.25 and "\n" in t and len(t.split()) < 120:
            return True

        return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_scores: bool = True,
        source: Optional[str] = None,
    ) -> List[Tuple[Document, float]] | List[Document]:

        if top_k is None:
            top_k = self.top_k

        where: Optional[Dict] = {"source": source} if source else None

        logger.info(f"🔍 retrieve strategy={self.strategy} top_k={top_k} fetch_k={self.fetch_k} source={source}")

        if self.strategy.lower() == "mmr":
            docs = self.vectorstore.mmr_search(
                query,
                k=min(self.fetch_k, 50),
                fetch_k=self.fetch_k,
                lambda_mult=self.mmr_lambda,
                where=where,
            )
            docs = [d for d in docs if not self._is_low_value_chunk(d.page_content)]
            docs = docs[:top_k]
            # placeholder score for MMR
            return [(d, 0.0) for d in docs] if with_scores else docs

        results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k, where=where)
        results = [(d, s) for (d, s) in results if not self._is_low_value_chunk(d.page_content)]
        results = self._dedupe(results)
        results.sort(key=lambda x: float(x[1]))
        final = results[:top_k]
        return final if with_scores else [d for d, _ in final]

    def format_context(self, documents: List[Tuple[Document, float]] | List[Document]) -> str:
        parts = []
        for i, item in enumerate(documents, 1):
            if isinstance(item, tuple):
                doc, score = item
            else:
                doc, score = item, None

            cleaned = self._strip_leading_junk(doc.page_content)
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            parts.append(f"[Document {i} - Source: {source}, Page: {page}]\n{cleaned}\n")
        return "\n".join(parts)

def get_retriever() -> Retriever:
    return Retriever()
