from typing import List, Tuple, Optional
from langchain_core.documents import Document
from app.embeddings.vectorstore import get_vectorstore
from utils.config_loader import config
from utils.logger import logger


class Retriever:
    """Handle document retrieval"""

    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.top_k = int(config.get("retrieval", "top_k", default=5))

        # Strategy: "mmr" or "similarity"
        self.strategy = config.get("retrieval", "strategy", default="mmr")

        # MMR params
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

    def _looks_like_definition(self, text: str) -> bool:
        t = (text or "").lower()
        return any(p in t for p in [
            "we define", "is defined as", "defined as", "stands for", "refers to", "is a method"
        ])

    def _is_low_value_chunk(self, text: str) -> bool:
        """
        Drop chunks that are usually unhelpful:
        - references/bibliography
        - author lists
        - numeric tables-only dumps

        BUT keep short chunks if they look like definitions.
        """
        t = (text or "").strip()
        if not t:
            return True

        # keep short chunks if definition-like
        if len(t) < 220 and self._looks_like_definition(t):
            return False

        # too short and not definition-like
        if len(t) < 120:
            return True

        head = t[:400].lower()

        if "references" in head or "bibliography" in head:
            return True

        # author list heuristic
        comma_ratio = t.count(",") / max(1, len(t))
        if comma_ratio > 0.02 and any(x in head for x in ["openai", "university", "google", "microsoft"]):
            return True

        # numeric-heavy table chunk
        digits = sum(ch.isdigit() for ch in t)
        digit_ratio = digits / max(1, len(t))
        if digit_ratio > 0.25 and "\n" in t and len(t.split()) < 120:
            return True

        return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_scores: bool = True
    ) -> List[Tuple[Document, float]] | List[Document]:

        if top_k is None:
            top_k = self.top_k

        logger.info(f"🔍 Retrieving strategy={self.strategy} top_k={top_k} fetch_k={self.fetch_k} query='{query}'")

        try:
            if self.strategy.lower() == "mmr":
                # MMR: return top_k, fetch_k is the candidate pool
                docs = self.vectorstore.mmr_search(
                    query,
                    k=top_k * 2,               # return a bit more, then filter
                    fetch_k=self.fetch_k,
                    lambda_mult=self.mmr_lambda
                )

                docs = [d for d in docs if not self._is_low_value_chunk(d.page_content)]
                docs = docs[:top_k]

                return [(d, 0.0) for d in docs] if with_scores else docs

            # Similarity fallback
            results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
            results = [(d, s) for (d, s) in results if not self._is_low_value_chunk(d.page_content)]
            results = self._dedupe(results)
            results.sort(key=lambda x: float(x[1]))

            final = results[:top_k]
            logger.info(f"✅ Retrieved {len(final)} docs after filtering/dedupe/sort")
            return final if with_scores else [d for d, _ in final]

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            raise

    def format_context(self, documents: List[Tuple[Document, float]] | List[Document]) -> str:
        context_parts = []
        for i, item in enumerate(documents, 1):
            if isinstance(item, tuple):
                doc, score = item
                score_text = f" (score: {score:.4f})" if score else ""
            else:
                doc = item
                score_text = ""

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            cleaned_text = self._strip_leading_junk(doc.page_content)

            context_parts.append(
                f"[Document {i} - Source: {source}, Page: {page}{score_text}]\n"
                f"{cleaned_text}\n"
            )
        return "\n".join(context_parts)


def get_retriever() -> Retriever:
    return Retriever()
