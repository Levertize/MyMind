# MyMind — Personal RAG Chatbot

Chatbot berbasis RAG (Retrieval-Augmented Generation) yang memungkinkan user untuk chat dengan dokumen pribadi mereka (PDF, TXT, Markdown, EPUB) menggunakan Gemini API.

---

## Project Structure

```
mymind/
├── ingestion/          # Baca & proses dokumen
│   ├── loader.py       # Support PDF, TXT, MD
│   └── chunker.py      # Chunking strategy
├── embeddings/
│   └── embedder.py     # Gemini Embedding API
├── storage/
│   └── vectorstore.py  # ChromaDB (local)
├── retrieval/
│   └── retriever.py    # Semantic search
├── llm/
│   └── generator.py    # Gemini generate answer
├── data/               # Folder dokumen user (gitignored)
├── docs/               # Dokumentasi tambahan
├── cli.py              # Entry point CLI
├── config.py           # Konfigurasi & env loader
├── .env.example        # Template env vars
└── requirements.txt
```

---

## Tech Stack

- **Language:** Python 3.11+
- **LLM & Embedding:** Google Gemini API (`google-generativeai`)
- **Vector DB:** ChromaDB (local, persistent)
- **Doc Parser:** PyMuPDF (PDF), built-in (TXT, MD)
- **Orchestration:** LangChain
- **Interface:** CLI (argparse), lalu FastAPI

---

## Environment Variables

```
GEMINI_API_KEY=your_api_key_here
CHROMA_DB_PATH=./chroma_db
DATA_DIR=./data
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

---

## Rules

### Code Style
- Gunakan Python type hints di semua fungsi
- Setiap modul punya docstring singkat di bagian atas
- Nama variabel & fungsi dalam bahasa Inggris
- Komentar boleh dalam bahasa Indonesia

### Arsitektur
- Setiap folder adalah modul terpisah dengan tanggung jawab tunggal (Single Responsibility)
- Jangan hardcode API key — selalu pakai `config.py` + `.env`
- Fungsi tidak boleh lebih dari 40 baris; pecah jika perlu
- Hindari circular import antar modul

### Gemini API
- Model embedding: `models/text-embedding-004`
- Model generate: `gemini-1.5-flash` (hemat quota)
- Selalu handle exception dari API call (rate limit, timeout)
- Batasi konteks yang dikirim ke LLM maksimal 5 chunk teratas

### Git
- Commit message pakai format: `feat:`, `fix:`, `refactor:`, `docs:`
- Jangan commit file `.env`, folder `data/`, dan `chroma_db/`
- Setiap fitur baru di branch terpisah

### Prioritas Development
1. **MVP dulu:** ingest → embed → store → retrieve → generate
2. Baru tambah fitur: conversation history, multi-format, web UI
- Jangan loncat ke fitur baru sebelum MVP berjalan end-to-end