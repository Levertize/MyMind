"""
Module to manage ChromaDB local vector database storage.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH

# Inisialisasi ChromaDB client persistent
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_or_create_collection(name: str = "mymind_documents"):
    """
    Mengambil atau membuat ChromaDB collection baru.
    """
    return client.get_or_create_collection(name=name)

def add_documents(
    collection_name: str, 
    texts: List[str], 
    embeddings: List[List[float]], 
    metadatas: List[Dict[str, Any]], 
    ids: List[str]
) -> None:
    """
    Menambahkan dokumen, embedding, dan metadata ke dalam ChromaDB.
    """
    collection = get_or_create_collection(collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def query_documents(
    collection_name: str, 
    query_embeddings: List[List[float]], 
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Mencari dokumen terdekat berdasarkan vektor query embedding.
    """
    collection = get_or_create_collection(collection_name)
    return collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )
