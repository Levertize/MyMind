"""
Module to generate answers using LangChain ChatGoogleGenerativeAI based on retrieved context.
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import GEMINI_API_KEY, GENERATION_MODEL

# Inisialisasi model LLM Chat Google Generative AI
chat_model = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

def generate_answer(
    query: str, 
    context_chunks: List[Dict[str, Any]], 
    history: List[Dict[str, str]] = None
) -> str:
    """
    Menghasilkan jawaban menggunakan ChatGoogleGenerativeAI berdasarkan query, konteks, dan riwayat percakapan.
    """
    # 1. Bangun prompt dengan menyertakan konteks
    context_text: str = ""
    for idx, chunk in enumerate(context_chunks):
        source = chunk.get("metadata", {}).get("source", "Unknown Source")
        context_text += f"\n--- CHUNK {idx+1} (Source: {source}) ---\n{chunk['text']}\n"

    system_prompt = (
        "Anda adalah MyMind, asisten chatbot pribadi yang membantu menjawab pertanyaan user berdasarkan "
        "konteks dokumen yang diberikan. Gunakan informasi dari dokumen di bawah ini untuk menjawab. "
        "Jika jawabannya tidak ada di dalam dokumen, katakan secara jujur bahwa Anda tidak mengetahuinya "
        "atau jawabannya tidak ditemukan di dokumen pribadi user.\n\n"
        "Dokumen Konteks:\n"
        f"{context_text}"
    )

    messages = [SystemMessage(content=system_prompt)]

    # 2. Bangun riwayat percakapan menggunakan struktur Message LangChain
    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

    # Tambahkan query saat ini
    messages.append(HumanMessage(content=query))

    try:
        # 3. Kirim ke model Gemini melalui LangChain
        response = chat_model.invoke(messages)
        return str(response.content)
    except Exception as e:
        raise RuntimeError(f"Gagal generate jawaban dari LangChain ChatGoogleGenAI: {str(e)}")
