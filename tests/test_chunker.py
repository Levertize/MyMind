"""
Unit tests for text chunker module (ingestion/chunker.py).
"""

from ingestion.chunker import split_text

def test_split_text_normal() -> None:
    """
    Menguji pemotongan teks standar dengan chunk_size dan chunk_overlap tertentu.
    """
    text = "Ini adalah sebuah teks contoh yang cukup panjang untuk dipotong oleh chunker."
    chunks = split_text(text, chunk_size=20, chunk_overlap=5)
    
    assert len(chunks) > 0
    for chunk in chunks:
        # Periksa panjang chunk tidak melebihi chunk_size
        assert len(chunk) <= 20

def test_split_text_empty() -> None:
    """
    Menguji input teks kosong pada splitter.
    """
    chunks = split_text("")
    assert chunks == []
