"""
Module for retrieving relevant context from vector database using semantic search.
"""

from typing import List, Dict, Any
from embeddings.embedder import get_embedding
from storage.vectorstore import query_documents
from config import MAX_CONTEXT_CHUNKS

def retrieve_context(query: str, collection_name: str = "mymind_documents") -> List[Dict[str, Any]]:
    """
    Mengambil potongan teks yang paling relevan dengan query user menggunakan similarity search.
    Mengembalikan maksimal 5 chunk teratas sesuai aturan.
    """
    # 1. Dapatkan embedding dari query user
    query_vector: List[float] = get_embedding(query)
    
    # 2. Cari di database ChromaDB
    results = query_documents(
        collection_name=collection_name,
        query_embeddings=[query_vector],
        n_results=MAX_CONTEXT_CHUNKS
    )
    
    # 3. Format hasil pencarian
    retrieved_chunks: List[Dict[str, Any]] = []
    
    if not results or "documents" not in results or not results["documents"][0]:
        return retrieved_chunks
        
    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(documents)
    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(documents)
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved_chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })
        
    return retrieved_chunks
