"""
FastAPI Server Backend for MyMind Personal RAG Chatbot.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import shutil

from ingestion.loader import load_document
from ingestion.chunker import split_text
from embeddings.embedder import get_embeddings
from storage.vectorstore import add_documents, delete_document, clear_vectorstore, list_documents
from llm.orchestrator import run_rag_pipeline
from config import DATA_DIR, GENERATION_MODEL

app = FastAPI(title="MyMind Personal RAG Chatbot")

# Model data request chat
class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

@app.get("/", response_class=HTMLResponse)
def get_index() -> HTMLResponse:
    """
    Menyajikan halaman HTML frontend utama.
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template frontend tidak ditemukan.")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Endpoint untuk menerima pesan dan mengembalikan jawaban RAG serta referensi dokumen.
    """
    try:
        # Jalankan pipeline RAG menggunakan orchestrator LangChain
        result = run_rag_pipeline(request.query, request.history)
        answer = result["answer"]
        context_chunks = result["sources"]
        
        # Format referensi dokumen
        sources = []
        for chunk in context_chunks:
            sources.append({
                "source": chunk.get("metadata", {}).get("source", "Unknown"),
                "distance": float(chunk.get("distance", 0.0))
            })
            
        return {
            "answer": answer,
            "sources": sources,
            "model": GENERATION_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
def ingest_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint untuk mengunggah berkas dokumen dan memprosesnya ke ChromaDB.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Jalankan pipeline ingest
        content = load_document(file_path)
        chunks = split_text(content)
        if not chunks:
            return {"status": "error", "message": "Dokumen kosong."}
            
        embeddings = get_embeddings(chunks)
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]
        
        add_documents(
            collection_name="mymind_documents",
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Berhasil memproses {len(chunks)} bagian dokumen."
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

class DeleteRequest(BaseModel):
    filename: str

@app.post("/api/documents/delete")
def delete_document_endpoint(request: DeleteRequest) -> Dict[str, Any]:
    """
    Endpoint untuk menghapus dokumen tertentu berdasarkan nama file dari ChromaDB.
    """
    try:
        deleted = delete_document("mymind_documents", request.filename)
        if deleted:
            file_path = os.path.join(DATA_DIR, request.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            return {"status": "success", "message": f"Dokumen {request.filename} berhasil dihapus."}
        else:
            raise HTTPException(status_code=404, detail=f"Dokumen {request.filename} tidak ditemukan.")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/database/clear")
def clear_database_endpoint() -> Dict[str, Any]:
    """
    Endpoint untuk mengosongkan seluruh database ChromaDB.
    """
    try:
        clear_vectorstore("mymind_documents")
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                file_path = os.path.join(DATA_DIR, f)
                if os.path.isfile(file_path) and f != ".gitkeep":
                    os.remove(file_path)
        return {"status": "success", "message": "Database berhasil dikosongkan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
def get_documents_endpoint() -> Dict[str, Any]:
    """
    Endpoint untuk mendapatkan daftar seluruh dokumen yang tersimpan di ChromaDB.
    """
    try:
        docs = list_documents("mymind_documents")
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
