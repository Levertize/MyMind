"""
Module to manage Chroma vector database storage using LangChain.
"""

from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from embeddings.embedder import embeddings_model
from config import CHROMA_DB_PATH

def get_vectorstore(collection_name: str = "mymind_documents") -> Chroma:
    """
    Mengambil instance Chroma vectorstore LangChain.
    """
    return Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings_model
    )

def add_documents(
    collection_name: str, 
    texts: List[str], 
    embeddings: List[List[float]], 
    metadatas: List[Dict[str, Any]], 
    ids: List[str]
) -> None:
    """
    Menambahkan dokumen ke ChromaDB menggunakan LangChain.
    """
    db = get_vectorstore(collection_name)
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
