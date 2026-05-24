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

def delete_document(collection_name: str, filename: str) -> bool:
    """
    Menghapus dokumen dari ChromaDB berdasarkan nama file.
    Mengembalikan True jika sukses menghapus, False jika tidak ada data yang cocok.
    """
    db = get_vectorstore(collection_name)
    collection = db._collection
    
    # Dapatkan semua chunk yang memiliki source == filename
    results = collection.get(where={"source": filename})
    ids = results.get("ids", [])
    
    if ids:
        collection.delete(ids=ids)
        return True
    return False

def clear_vectorstore(collection_name: str) -> None:
    """
    Menghapus seluruh koleksi data di ChromaDB (clear database).
    """
    db = get_vectorstore(collection_name)
    db.delete_collection()

def list_documents(collection_name: str) -> List[str]:
    """
    Mengambil daftar nama file dokumen unik yang terindeks di ChromaDB.
    """
    db = get_vectorstore(collection_name)
    collection = db._collection
    
    # Ambil semua metadata di koleksi
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
    
    # Ekstrak nama file unik
    sources = set()
    for meta in metadatas:
        if meta and "source" in meta:
            sources.add(meta["source"])
            
    return sorted(list(sources))


