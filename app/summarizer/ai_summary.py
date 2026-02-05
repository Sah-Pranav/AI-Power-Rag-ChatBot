# app/summarizer/ai_summary.py

from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from app.summarizer.llm_factory import get_llm
from app.retriever.query import get_retriever
from utils.logger import logger

class RAGPipeline:
    """Complete RAG pipeline: Retrieval + Generation"""
    
    def __init__(self):
        self.llm = get_llm()
        self.retriever = get_retriever()
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup the prompt template"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. Answer ONLY based on the provided documents
2. If the answer is not in the documents, clearly state: "I could not find relevant information to answer your question in the provided documents."
3. Cite the relevant sources in your answer
4. Be precise and professional
5. ALWAYS answer in English only - never mix languages

Here are the relevant documents:

{context}"""),
            ("human", "{question}")
        ])
    
    def query(
        self, 
        question: str, 
        top_k: int = None,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete RAG query: retrieve + generate answer
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"❓ Processing question: '{question}'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                question, 
                top_k=top_k,
                with_scores=True
            )
            
            if not retrieved_docs:
                logger.warning("⚠️ No relevant documents found")
                return {
                    "answer": "I could not find relevant information to answer your question.",
                    "sources": [],
                    "retrieved_docs": 0
                }
            
            # Step 2: Format context
            context = self.retriever.format_context(retrieved_docs)
            
            # Step 3: Generate answer using LLM
            logger.info("🤖 Generating answer with LLM...")
            
            chain = self.prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            answer = response.content
            
            # Step 4: Prepare response
            result = {
                "answer": answer,
                "retrieved_docs": len(retrieved_docs)
            }
            
            if return_sources:
                sources = []
                for doc, score in retrieved_docs:
                    sources.append({
                        "source": doc.metadata.get('source', 'Unknown'),
                        "page": doc.metadata.get('page', 'N/A'),
                        "relevance": round(1 - score, 3),
                        "content_preview": doc.page_content[:200] + "..."
                    })
                result["sources"] = sources
            
            logger.info("✅ Answer generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise

# Global instance
def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance"""
    return RAGPipeline()

# app/summarizer/ai_summary.py

# from typing import Dict, List
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from app.summarizer.llm_factory import get_llm
# from app.retriever.query import get_retriever
# from utils.logger import logger

# class RAGPipeline:
#     """Complete RAG pipeline: Retrieval + Generation"""
    
#     def __init__(self):
#         self.llm = get_llm()
#         self.retriever = get_retriever()
#         self._setup_prompt()
    
#     def _setup_prompt(self):
#         """Setup the prompt template - UPDATED FOR ENGLISH"""
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a helpful AI assistant that answers questions based on provided documents.

# CRITICAL RULES:
# 1. Answer ONLY based on the provided documents below
# 2. If the answer is not in the documents, clearly state: "I could not find relevant information to answer your question in the provided documents."
# 3. Cite the relevant sources in your answer
# 4. Be precise, professional, and factual
# 5. ALWAYS answer in English only - never mix languages
# 6. Use clear, simple language

# Here are the relevant documents:

# {context}"""),
#             ("human", "{question}")
#         ])
    
#     def query(
#         self, 
#         question: str, 
#         top_k: int = None,
#         return_sources: bool = True
#     ) -> Dict:
#         """
#         Complete RAG query: retrieve + generate answer
        
#         Args:
#             question: User question
#             top_k: Number of documents to retrieve
#             return_sources: Whether to return source documents
            
#         Returns:
#             Dict with answer, sources, and metadata
#         """
#         logger.info(f"❓ Processing question: '{question}'")
        
#         try:
#             # Step 1: Retrieve relevant documents
#             retrieved_docs = self.retriever.retrieve(
#                 question, 
#                 top_k=top_k,
#                 with_scores=True
#             )
            
#             if not retrieved_docs:
#                 logger.warning("⚠️ No relevant documents found")
#                 return {
#                     "answer": "I could not find relevant information to answer your question in the provided documents.",
#                     "sources": [],
#                     "retrieved_docs": 0
#                 }
            
#             # Step 2: Format context
#             context = self.retriever.format_context(retrieved_docs)
            
#             # Step 3: Generate answer using LLM
#             logger.info("🤖 Generating answer with LLM...")
            
#             chain = self.prompt | self.llm
#             response = chain.invoke({
#                 "context": context,
#                 "question": question
#             })
            
#             answer = response.content
            
#             # Step 4: Prepare response
#             result = {
#                 "answer": answer,
#                 "retrieved_docs": len(retrieved_docs)
#             }
            
#             if return_sources:
#                 sources = []
#                 for doc, score in retrieved_docs:
#                     sources.append({
#                         "source": doc.metadata.get('source', 'Unknown'),
#                         "page": doc.metadata.get('page', 'N/A'),
#                         "relevance": round(1 - score, 3),
#                         "content_preview": doc.page_content[:200] + "..."
#                     })
#                 result["sources"] = sources
            
#             logger.info("✅ Answer generated successfully")
#             return result
            
#         except Exception as e:
#             logger.error(f"❌ Query failed: {e}")
#             raise

# # Global instance
# def get_rag_pipeline() -> RAGPipeline:
#     """Get RAG pipeline instance"""
#     return RAGPipeline()