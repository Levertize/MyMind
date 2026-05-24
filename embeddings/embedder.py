"""
Module to generate text embeddings using Gemini Embedding API.
"""

from typing import List
import google.generativeai as genai
from config import GEMINI_API_KEY, EMBEDDING_MODEL

# Konfigurasi Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text: str) -> List[float]:
    """
    Mendapatkan representasi embedding (vektor) dari satu teks.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY belum dikonfigurasi di file .env")
        
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        # Menangani error API (rate limit, timeout, dll)
        raise RuntimeError(f"Gagal mendapatkan embedding dari Gemini API: {str(e)}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Mendapatkan representasi embedding (vektor) untuk beberapa teks sekaligus.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY belum dikonfigurasi di file .env")
        
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Gagal mendapatkan embeddings dari Gemini API: {str(e)}")
