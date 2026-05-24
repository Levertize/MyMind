"""
Module to load documents from various formats (PDF, TXT, MD, EPUB).
"""

from typing import List, Dict, Any
import fitz  # PyMuPDF untuk PDF
import os

def load_text_file(file_path: str) -> str:
    """
    Membaca isi dari file teks (.txt atau .md).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf_file(file_path: str) -> str:
    """
    Membaca dan mengekstrak teks dari file PDF menggunakan PyMuPDF.
    """
    text: str = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def load_epub_file(file_path: str) -> str:
    """
    Membaca dan mengekstrak teks bersih dari file EPUB menggunakan ebooklib dan BeautifulSoup.
    """
    from bs4 import BeautifulSoup
    import ebooklib
    from ebooklib import epub
    import warnings

    # Abaikan warning dari library pihak ketiga (ebooklib)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(file_path)
        
    text_content: List[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text().strip()
        if text:
            text_content.append(text)
            
    return "\n\n".join(text_content)

def load_document(file_path: str) -> str:
    """
    Menentukan format dokumen dan memuat isinya.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
        
    ext: str = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        return load_text_file(file_path)
    elif ext == ".pdf":
        return load_pdf_file(file_path)
    elif ext == ".epub":
        return load_epub_file(file_path)
    else:
        raise ValueError(f"Format file {ext} tidak didukung.")
