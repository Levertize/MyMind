"""
Module to generate text embeddings using LangChain GoogleGenAIEmbeddings.
"""

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, EMBEDDING_MODEL

# Inisialisasi embeddings model LangChain
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

def get_embedding(text: str) -> List[float]:
    """
    Mendapatkan representasi embedding (vektor) dari satu teks menggunakan LangChain.
    """
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        raise RuntimeError(f"Gagal mendapatkan embedding dari LangChain: {str(e)}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Mendapatkan representasi embedding (vektor) untuk beberapa teks sekaligus menggunakan LangChain.
    """
    try:
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(f"Gagal mendapatkan embeddings dari LangChain: {str(e)}")
