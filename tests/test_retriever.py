"""
Unit tests for retrieval module (retrieval/retriever.py).
"""

from typing import List
from unittest.mock import patch, MagicMock
from retrieval.retriever import retrieve_context

class MockDoc:
    """
    Mock class untuk objek Document LangChain.
    """
    def __init__(self, content: str, metadata: dict) -> None:
        self.page_content = content
        self.metadata = metadata

@patch("retrieval.retriever.get_vectorstore")
def test_retrieve_context_success(mock_get_vectorstore: MagicMock) -> None:
    """
    Menguji retrieve_context ketika database mengembalikan data.
    """
    mock_db = MagicMock()
    # Mock data dokumen dan skor
    doc1 = MockDoc("Konten dokumen A", {"source": "docA.txt"})
    doc2 = MockDoc("Konten dokumen B", {"source": "docB.txt"})
    
    mock_db.similarity_search_with_score.return_value = [
        (doc1, 0.35),
        (doc2, 0.45)
    ]
    mock_get_vectorstore.return_value = mock_db

    results = retrieve_context("query testing")
    
    assert len(results) == 2
    assert results[0]["text"] == "Konten dokumen A"
    assert results[0]["metadata"]["source"] == "docA.txt"
    assert results[0]["distance"] == 0.35
    assert results[1]["text"] == "Konten dokumen B"
    assert results[1]["distance"] == 0.45

@patch("retrieval.retriever.get_vectorstore")
def test_retrieve_context_empty(mock_get_vectorstore: MagicMock) -> None:
    """
    Menguji retrieve_context ketika database kosong.
    """
    mock_db = MagicMock()
    mock_db.similarity_search_with_score.return_value = []
    mock_get_vectorstore.return_value = mock_db

    results = retrieve_context("query testing")
    assert results == []
