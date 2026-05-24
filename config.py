"""
Module to load and manage configuration settings for MyMind RAG Chatbot.
"""

import os
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

# Konfigurasi API dan path
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR: str = os.getenv("DATA_DIR", "./data")

# Konfigurasi Chunker
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# Konfigurasi Model Gemini
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")
MAX_CONTEXT_CHUNKS: int = 5

