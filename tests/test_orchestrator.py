"""
Unit tests for LangChain RAG orchestrator module (llm/orchestrator.py).
"""

from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from llm.orchestrator import format_history, format_context, run_rag_pipeline

def test_format_history() -> None:
    """
    Menguji konversi riwayat percakapan dict ke object message LangChain.
    """
    history = [
        {"role": "user", "content": "Halo asisten"},
        {"role": "model", "content": "Halo! Ada yang bisa dibantu?"}
    ]
    messages = format_history(history)
    
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Halo asisten"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Halo! Ada yang bisa dibantu?"

def test_format_context() -> None:
    """
    Menguji penyusunan teks konteks dari chunk data.
    """
    chunks = [
        {"text": "Isi dokumen ke-1.", "metadata": {"source": "doc1.txt"}},
        {"text": "Isi dokumen ke-2.", "metadata": {"source": "doc2.txt"}}
    ]
    context_text = format_context(chunks)
    
    assert "doc1.txt" in context_text
    assert "Isi dokumen ke-1." in context_text
    assert "doc2.txt" in context_text
    assert "Isi dokumen ke-2." in context_text

@patch("llm.orchestrator.retrieve_context")
@patch("llm.orchestrator.rag_chain")
def test_run_rag_pipeline_success(mock_rag_chain: MagicMock, mock_retrieve: MagicMock) -> None:
    """
    Menguji alur pipeline run_rag_pipeline ketika pemanggilan sukses.
    """
    # Mock data retrieval
    mock_retrieve.return_value = [
        {"text": "Budi hobi main catur.", "metadata": {"source": "test.txt"}, "distance": 0.1}
    ]
    # Mock chain response
    mock_rag_chain.invoke.return_value = "Hobi Budi adalah catur."
    
    result = run_rag_pipeline(query="Apa hobi Budi?")
    
    assert result["answer"] == "Hobi Budi adalah catur."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["text"] == "Budi hobi main catur."
    
    mock_retrieve.assert_called_once_with("Apa hobi Budi?")
    mock_rag_chain.invoke.assert_called_once()

@patch("llm.orchestrator.retrieve_context")
@patch("llm.orchestrator.rag_chain")
def test_run_rag_pipeline_failure(mock_rag_chain: MagicMock, mock_retrieve: MagicMock) -> None:
    """
    Menguji alur pipeline run_rag_pipeline ketika API LLM mengalami kegagalan.
    """
    mock_retrieve.return_value = []
    # Mock error saat pemanggilan model
    mock_rag_chain.invoke.side_effect = Exception("API Quota Exceeded")
    
    result = run_rag_pipeline(query="Gagal?")
    
    assert "Maaf, saat ini layanan Gemini API sedang sibuk" in result["answer"]
    assert "API Quota Exceeded" in result["answer"]
    assert result["sources"] == []

