"""
Module for chunking loaded document text into smaller segments.
"""

from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Membagi teks menjadi potongan-potongan kecil (chunks) dengan overlap tertentu.
    Menggunakan strategi pemisahan sederhana berdasarkan karakter/kata.
    """
    # Placeholder implementasi pemisahan teks sederhana
    chunks: List[str] = []
    start: int = 0
    text_len: int = len(text)
    
    if text_len == 0:
        return chunks

    while start < text_len:
        end: int = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        
    return chunks
