"""
Entry point CLI for MyMind Personal RAG Chatbot.
"""

import argparse
import sys
import os
from typing import List, Dict, Any
from ingestion.loader import load_document
from ingestion.chunker import split_text
from embeddings.embedder import get_embeddings
from storage.vectorstore import add_documents
from llm.orchestrator import run_rag_pipeline
from config import DATA_DIR, GENERATION_MODEL

def run_ingest(file_path: str) -> None:
    """
    Melakukan proses ingestion dokumen: Load -> Chunk -> Embed -> Store.
    """
    print(f"[*] Memulai proses ingest untuk: {file_path}")
    
    # 1. Load document
    try:
        content = load_document(file_path)
        print(f"[+] Dokumen berhasil dimuat. Panjang karakter: {len(content)}")
    except Exception as e:
        print(f"[-] Gagal memuat dokumen: {str(e)}")
        return

    # 2. Chunk text
    chunks = split_text(content)
    print(f"[+] Teks dibagi menjadi {len(chunks)} chunks.")
    if not chunks:
        print("[-] Tidak ada teks untuk diproses.")
        return

    # 3. Generate embeddings
    print("[*] Menghubungi Gemini API untuk mendapatkan embeddings...")
    try:
        embeddings = get_embeddings(chunks)
        print("[+] Embeddings berhasil didapatkan.")
    except Exception as e:
        print(f"[-] Gagal mendapatkan embeddings: {str(e)}")
        return

    # 4. Store in Vector DB
    print("[*] Menyimpan ke ChromaDB...")
    try:
        filename = os.path.basename(file_path)
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        
        add_documents(
            collection_name="mymind_documents",
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[+] Sukses menyimpan {len(chunks)} chunk ke database local.")
    except Exception as e:
        print(f"[-] Gagal menyimpan ke database: {str(e)}")

def run_query(query: str) -> None:
    """
    Melakukan proses QA dengan retrieval menggunakan LangChain Orchestration.
    """
    print(f"[*] Query: '{query}'")
    print(f"[*] Menghasilkan jawaban menggunakan RAG pipeline LangChain ({GENERATION_MODEL})...")
    try:
        result = run_rag_pipeline(query)
        answer = result["answer"]
        context_chunks = result["sources"]
    except Exception as e:
        print(f"[-] Gagal memproses query: {str(e)}")
        return

    if context_chunks:
        # Tampilkan sumber referensi yang ditemukan
        print("\n--- Referensi Dokumen Terdekat ---")
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            distance = chunk.get("distance", 0.0)
            print(f"[{i+1}] {source} (Score distance: {distance:.4f})")
        print("---------------------------------\n")
    else:
        print("[-] Tidak ditemukan dokumen referensi terdekat di database.")

    print("\n=== JAWABAN MYMIND ===")
    print(answer)
    print("======================\n")

def run_chat() -> None:
    """
    Menjalankan sesi interactive chat CLI menggunakan LangChain Orchestration.
    """
    print("====================================================")
    print("      Selamat datang di Sesi Chat Interaktif MyMind")
    print("   (Ketik 'exit' atau 'quit' untuk mengakhiri obrolan)")
    print("====================================================\n")
    
    history: List[Dict[str, str]] = []
    
    while True:
        try:
            query = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[*] Mengakhiri sesi chat. Sampai jumpa!")
            break
            
        if not query:
            continue
            
        if query.lower() in ["exit", "quit"]:
            print("[*] Mengakhiri sesi chat. Sampai jumpa!")
            break
            
        try:
            result = run_rag_pipeline(query, history)
            answer = result["answer"]
            print(f"\nMyMind > {answer}\n")
            
            history.append({"role": "user", "content": query})
            history.append({"role": "model", "content": answer})
            
            # Batasi riwayat agar tidak terlalu panjang (misal, 10 pesan terakhir)
            if len(history) > 10:
                history = history[-10:]
        except Exception as e:
            print(f"\n[-] Gagal memproses obrolan: {str(e)}\n")

def main() -> None:
    """
    Main entry point untuk CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="MyMind — Personal RAG Chatbot CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Perintah CLI")
    
    # Subparser untuk Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest file dokumen ke database")
    ingest_parser.add_argument("file_path", type=str, help="Path ke file dokumen (PDF, TXT, MD)")
    
    # Subparser untuk Query
    query_parser = subparsers.add_parser("query", help="Tanyakan sesuatu ke dokumen pribadi Anda")
    query_parser.add_argument("prompt", type=str, help="Pertanyaan untuk chatbot")
    
    # Subparser untuk Chat
    subparsers.add_parser("chat", help="Mulai sesi obrolan interaktif dengan memori")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        run_ingest(args.file_path)
    elif args.command == "query":
        run_query(args.prompt)
    elif args.command == "chat":
        run_chat()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
