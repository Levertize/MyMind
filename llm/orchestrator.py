"""
Module for orchestrating the RAG pipeline using LangChain Expression Language (LCEL).
"""

from typing import List, Dict, Any
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from retrieval.retriever import retrieve_context
from llm.generator import chat_model

def format_history(history: List[Dict[str, str]] = None) -> List[BaseMessage]:
    """
    Mengonversi riwayat percakapan dari format dict ke list BaseMessage LangChain.
    """
    messages: List[BaseMessage] = []
    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
    return messages

def format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Memformat context chunks menjadi string terstruktur untuk system prompt.
    """
    context_text = ""
    for idx, chunk in enumerate(context_chunks):
        source = chunk.get("metadata", {}).get("source", "Unknown Source")
        context_text += f"\n--- CHUNK {idx+1} (Source: {source}) ---\n{chunk['text']}\n"
    return context_text

def create_rag_chain() -> Runnable:
    """
    Membuat chain LCEL RAG menggunakan prompt template, LLM, dan output parser.
    """
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="system_message"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    return prompt | chat_model | StrOutputParser()

# Inisialisasi chain orchestration dengan automatic retry (5 kali percobaan)
rag_chain = create_rag_chain().with_retry(stop_after_attempt=5)

def run_rag_pipeline(query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Menjalankan pipeline RAG menggunakan chain orchestration LangChain.
    """
    # 1. Retrieve konteks terdekat (dengan penanganan error gracefully)
    try:
        context_chunks = retrieve_context(query)
        context_text = format_context(context_chunks)
    except Exception:
        context_chunks = []
        context_text = "Dokumen konteks tidak dapat dimuat."
    
    # 2. Siapkan instruksi system dengan dokumen
    system_text = (
        "Anda adalah MyMind, asisten chatbot pribadi yang membantu menjawab pertanyaan user berdasarkan "
        "konteks dokumen yang diberikan. Gunakan informasi dari dokumen di bawah ini untuk menjawab. "
        "Jika jawabannya tidak ada di dalam dokumen, katakan secara jujur bahwa Anda tidak mengetahuinya "
        "atau jawabannya tidak ditemukan di dokumen pribadi user.\n\n"
        "Dokumen Konteks:\n"
        f"{context_text}"
    )
    system_messages = [SystemMessage(content=system_text)]
    
    # 3. Jalankan chain dengan penanganan exception rate limit, timeout, dsb.
    try:
        response_text = rag_chain.invoke({
            "system_message": system_messages,
            "chat_history": format_history(history),
            "query": query
        })
    except Exception as e:
        response_text = (
            "Maaf, saat ini layanan Gemini API sedang sibuk atau mengalami kendala teknis. "
            f"Silakan coba beberapa saat lagi. (Detail: {str(e)})"
        )
    
    return {
        "answer": response_text,
        "sources": context_chunks
    }

