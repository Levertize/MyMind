"""
Module to generate answers using Gemini LLM API based on retrieved context.
"""

from typing import List, Dict, Any
import google.generativeai as genai
from config import GEMINI_API_KEY, GENERATION_MODEL

# Konfigurasi Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_answer(
    query: str, 
    context_chunks: List[Dict[str, Any]], 
    history: List[Dict[str, str]] = None
) -> str:
    """
    Menghasilkan jawaban menggunakan Gemini model berdasarkan query, konteks, dan riwayat percakapan.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY belum dikonfigurasi di file .env")

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

    try:
        # 2. Inisialisasi model
        model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=system_prompt
        )
        
        # 3. Bangun parameter isi (contents) yang menyertakan riwayat percakapan
        contents = []
        if history:
            for turn in history:
                contents.append({
                    "role": "user" if turn["role"] == "user" else "model",
                    "parts": [turn["content"]]
                })
        
        # Tambahkan query saat ini
        contents.append({
            "role": "user",
            "parts": [query]
        })
        
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        # Menangani exception rate limit, timeout, dll dari API call
        raise RuntimeError(f"Gagal generate jawaban dari Gemini API: {str(e)}")
