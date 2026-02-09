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

    # ----------------------------------------------------
    # Cleaning helpers
    # ----------------------------------------------------

    def _strip_leading_junk(self, text: str) -> str:
        """Remove leading dots or broken chunk starts."""
        t = (text or "").lstrip()
        while t.startswith("...") or t.startswith("…") or t.startswith("."):
            t = t.lstrip(".… ").lstrip()
        return t

    def _dedupe(self, docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove near-duplicate chunks (same source/page/leading text)."""
        seen = set()
        out: List[Tuple[Document, float]] = []

        for doc, score in docs_with_scores:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content[:120]
            )
            if key in seen:
                continue
            seen.add(key)
            out.append((doc, score))

        return out

    def _is_low_value_chunk(self, text: str) -> bool:
        """
        Heuristic filter to drop chunks that are usually unhelpful:
        - references
        - bibliography
        - author lists
        - numeric tables
        """
        t = (text or "").strip()

        if len(t) < 200:
            return True

        head = t[:300].lower()

        # References or bibliography
        if "references" in head or "bibliography" in head:
            return True

        # Author list heuristic
        comma_ratio = t.count(",") / max(1, len(t))
        if comma_ratio > 0.02 and (
            "openai" in head or
            "university" in head or
            "google" in head or
            "microsoft" in head
        ):
            return True

        # Numeric heavy table chunks
        digits = sum(ch.isdigit() for ch in t)
        digit_ratio = digits / max(1, len(t))
        if digit_ratio > 0.25 and "\n" in t and len(t.split()) < 120:
            return True

        return False

    # ----------------------------------------------------
    # Retrieval
    # ----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_scores: bool = True
    ) -> List[Tuple[Document, float]] | List[Document]:
        """
        Retrieve relevant documents for a query.
        """

        if top_k is None:
            top_k = self.top_k

        logger.info(
            f"🔍 Retrieving strategy={self.strategy} top_k={top_k} "
            f"fetch_k={self.fetch_k} query='{query}'"
        )

        try:
            # --- MMR retrieval (diverse results) ---
            if self.strategy.lower() == "mmr":

                docs = self.vectorstore.mmr_search(
                    query,
                    k=min(self.fetch_k, 50),
                    fetch_k=self.fetch_k,
                    lambda_mult=self.mmr_lambda
                )

                # Filter low-value chunks
                docs = [d for d in docs if not self._is_low_value_chunk(d.page_content)]

                # Keep top_k after filtering
                docs = docs[:top_k]

                return [(d, 0.0) for d in docs] if with_scores else docs

            # --- Similarity retrieval (fallback) ---
            results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)

            # Drop low-value chunks
            results = [(d, s) for (d, s) in results if not self._is_low_value_chunk(d.page_content)]

            # Dedupe & sort (lower distance = better)
            results = self._dedupe(results)
            results.sort(key=lambda x: float(x[1]))

            final = results[:top_k]

            logger.info(f"✅ Retrieved {len(final)} docs after filtering/dedupe/sort")
            return final if with_scores else [d for d, _ in final]

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            raise

    # ----------------------------------------------------
    # Context Formatting
    # ----------------------------------------------------

    def format_context(
        self,
        documents: List[Tuple[Document, float]] | List[Document]
    ) -> str:
        """Format retrieved documents into context string."""
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
