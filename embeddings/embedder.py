"""
Module to generate text embeddings using LangChain GoogleGenAIEmbeddings.
"""

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, EMBEDDING_MODEL
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Inisialisasi embeddings model LangChain
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def _embed_query_with_retry(text: str) -> List[float]:
    """
    Memanggil API embed query dengan mekanisme retry exponential backoff.
    """
    return embeddings_model.embed_query(text)

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def _embed_documents_with_retry(texts: List[str]) -> List[List[float]]:
    """
    Memanggil API embed documents dengan mekanisme retry exponential backoff.
    """
    return embeddings_model.embed_documents(texts)

def get_embedding(text: str) -> List[float]:
    """
    Mendapatkan representasi embedding (vektor) dari satu teks menggunakan LangChain.
    """
    try:
        return _embed_query_with_retry(text)
    except Exception as e:
        raise RuntimeError(f"Gagal mendapatkan embedding dari LangChain setelah retry: {str(e)}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Mendapatkan representasi embedding (vektor) untuk beberapa teks sekaligus menggunakan LangChain.
    """
    try:
        return _embed_documents_with_retry(texts)
    except Exception as e:
        raise RuntimeError(f"Gagal mendapatkan embeddings dari LangChain setelah retry: {str(e)}")

