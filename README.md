# MyMind - Personal RAG Chatbot


[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Orchestration](https://img.shields.io/badge/orchestration-LangChain-green.svg)](https://python.langchain.com)
[![AI Model](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)](https://ai.google.dev)
[![Vector DB](https://img.shields.io/badge/vector%20db-ChromaDB-blueviolet.svg)](https://trychroma.com)
[![API Server](https://img.shields.io/badge/api-FastAPI-teal.svg)](https://fastapi.tiangolo.com)
[![Test Suite](https://img.shields.io/badge/tests-pytest-yellow.svg)](https://pytest.org)

Chatbot berbasis RAG (Retrieval-Augmented Generation) yang memungkinkan pengguna untuk melakukan pencarian semantik dan mengobrol secara cerdas dengan dokumen pribadi (PDF, TXT, Markdown, EPUB) menggunakan Google Gemini API dan diorkestrasikan oleh LangChain.

---

## Spesifikasi Teknologi yang Digunakan

Berikut adalah spesifikasi detail dari komponen dan pustaka yang telah diintegrasikan pada proyek ini:

### 1. Model AI (Google Gemini API)
- Model Embedding: models/gemini-embedding-001 (digunakan untuk merepresentasikan dokumen ke dalam bentuk vektor numerik).
- Model Generator Default: gemini-3.5-flash (digunakan untuk merumuskan jawaban berdasarkan konteks dokumen).
- Model Generator Cadangan: models/gemini-2.0-flash / models/gemini-2.5-flash (digunakan sebagai alternatif saat kuota free-tier habis).

### 2. Pustaka Python & Dependensi Utama
- langchain-google-genai (GoogleGenerativeAIEmbeddings & ChatGoogleGenerativeAI): Menghubungkan aplikasi ke API Gemini dengan standar LangChain.
- langchain-core: Menyediakan abstraksi Runnable untuk penyusunan RAG chain terpadu dengan LCEL (LangChain Expression Language).
- chromadb: Database vektor lokal yang menyimpan dokumen hasil pemotongan beserta representasi embedding-nya.
- fastapi & uvicorn: Menyediakan server web lokal dan REST API endpoint untuk Web UI.
- tenacity: Mekanisme otomatisasi retry dengan exponential backoff untuk ketahanan API terhadap rate limit (429).
- pymupdf (fitz): Parser berkas PDF berkinerja tinggi.
- ebooklib & beautifulsoup4: Pengekstrak konten teks bersih dari berkas EPUB.
- pytest: Framework pengujian unit dan integrasi otomatis.
- python-dotenv: Pemuat variabel lingkungan dari file .env.

---

## Fitur Utama

- Multi-Format Ingestion: Mendukung ekstraksi dokumen dari berbagai format secara bersih (.pdf, .txt, .md, .epub).
- LangChain Orchestration: Pipeline RAG terpadu menggunakan LangChain Expression Language (LCEL) untuk menggabungkan modul pencarian dokumen, pemformatan konteks, pencocokan riwayat pesan, dan LLM generator secara otomatis.
- Gemini API Robustness (Anti-Crash): Dilengkapi dengan mekanisme retry otomatis menggunakan exponential backoff (via tenacity & LangChain retry) untuk meminimalisir kegagalan akibat batas kecepatan request (429 Rate Limit) serta penanganan error secara anggun.
- Interactive Deletion & Clearing: Fitur untuk menghapus dokumen spesifik dari database lokal (serta menghapus file fisik di penyimpanan) atau mengosongkan seluruh database sekaligus.
- Document List: Daftar file dokumen yang telah sukses terindeks di ChromaDB otomatis dimuat pada panel Web UI saat pertama kali dibuka.
- Interactive Memory Chat: Mendukung sesi obrolan interaktif yang mengingat konteks percakapan sebelumnya baik via CLI maupun Web UI.
- Modern Responsive Web UI: Antarmuka web minimalis bernuansa dark-mode modern yang responsif dan dilengkapi animasi halus.

---

## Konfigurasi Environment (.env)

Buat file .env di direktori utama proyek Anda dan isi konfigurasi berikut:

```env
GEMINI_API_KEY=your_api_key_here
CHROMA_DB_PATH=./chroma_db
DATA_DIR=./data
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Ganti model ini jika Anda terkena batas kuota limit free-tier (429 Rate Limit)
# Model alternatif: models/gemini-2.0-flash, models/gemini-2.5-flash, gemini-3.5-flash
GENERATION_MODEL=models/gemini-2.0-flash
EMBEDDING_MODEL=models/gemini-embedding-001
```

---

## Panduan Memulai

### 1. Instalasi Dependensi
Buat virtual environment dan instal semua dependensi yang tertera pada requirements.txt:
```bash
# Membuat virtual environment
python -m venv venv
source venv/Scripts/activate # Untuk Windows (Powershell/CMD)

# Instalasi paket python
pip install -r requirements.txt
```

### 2. Penggunaan CLI (Command Line Interface)
Aplikasi MyMind dilengkapi perintah CLI yang lengkap untuk mengelola chatbot langsung dari terminal Anda:

* Ingest (Unggah Dokumen)
  ```bash
  python cli.py ingest data/dokumen_saya.txt
  ```
* Query (Pertanyaan Sekali Jawab)
  ```bash
  python cli.py query "Apa isi dokumen tentang catur?"
  ```
* Chat (Obrolan Interaktif dengan Memori)
  ```bash
  python cli.py chat
  ```
* Delete (Hapus Dokumen Spesifik)
  ```bash
  python cli.py delete dokumen_saya.txt
  ```
* Clear (Kosongkan Database)
  ```bash
  python cli.py clear
  ```

### 3. Penggunaan Web UI (FastAPI)
Untuk menjalankan server lokal dan mengakses Web UI interaktif:
```bash
python app.py
```
Setelah server aktif, buka browser Anda dan navigasikan ke alamat:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Cara Mengatasi Limit Kuota (429 API Rate Limit)

Jika Anda mendapatkan respon error:
> *Maaf, saat ini layanan Gemini API sedang sibuk atau mengalami kendala teknis (Detail: 429 RESOURCE_EXHAUSTED)...*

Artinya batas kuota free-tier harian Anda untuk model saat ini telah habis. Anda dapat mengatasinya secara instan dengan mengganti model generator ke model Gemini lain di file .env Anda (karena setiap model memiliki kuota terpisah):

```env
# Mengubah dari gemini-3.5-flash ke gemini-2.0-flash
GENERATION_MODEL=models/gemini-2.0-flash
```
Setelah mengubah .env, cukup restart server python app.py Anda.

---

## Menjalankan Automated Tests

Proyek ini dilengkapi dengan modul pengujian otomatis menggunakan pytest untuk memastikan seluruh fitur (loader, chunker, retriever, dan orchestrator) berjalan dengan benar.

Jalankan perintah berikut di direktori utama:
```bash
python -m pytest
```
*Catatan: Semua pengujian LLM dan database telah di-mock agar tidak mengonsumsi kuota API key riil Anda selama test berjalan.*
