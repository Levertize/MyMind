"""
Module for retrieving relevant context from vector database using LangChain retriever.
"""

from typing import List, Dict, Any
from storage.vectorstore import get_vectorstore
from config import MAX_CONTEXT_CHUNKS

def retrieve_context(query: str, collection_name: str = "mymind_documents") -> List[Dict[str, Any]]:
    """
    Mengambil potongan teks yang paling relevan dengan query user menggunakan similarity search LangChain.
    Mengembalikan maksimal 5 chunk teratas sesuai aturan.
    """
    db = get_vectorstore(collection_name)
    
    # Lakukan similarity search dengan score (distance)
    docs_with_scores = db.similarity_search_with_score(query, k=MAX_CONTEXT_CHUNKS)
    
    retrieved_chunks: List[Dict[str, Any]] = []
    for doc, score in docs_with_scores:
        retrieved_chunks.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "distance": float(score)
        })
        
    return retrieved_chunks
