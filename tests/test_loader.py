"""
Unit tests for document loader module (ingestion/loader.py).
"""

import pytest
from unittest.mock import patch, mock_open, MagicMock
from ingestion.loader import load_text_file, load_pdf_file, load_epub_file, load_document

def test_load_text_file() -> None:
    """
    Menguji fungsi load_text_file dengan open mock.
    """
    mock_data = "Isi berkas dokumen testing."
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = load_text_file("dummy.txt")
        assert result == mock_data

@patch("fitz.open")
def test_load_pdf_file(mock_fitz_open: MagicMock) -> None:
    """
    Menguji fungsi load_pdf_file dengan pymupdf mock.
    """
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Halaman pertama teks."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = " Halaman kedua teks."
    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_fitz_open.return_value.__enter__.return_value = mock_doc

    result = load_pdf_file("dummy.pdf")
    assert result == "Halaman pertama teks. Halaman kedua teks."

@patch("ebooklib.epub.read_epub")
@patch("bs4.BeautifulSoup")
def test_load_epub_file(mock_bs4: MagicMock, mock_read_epub: MagicMock) -> None:
    """
    Menguji fungsi load_epub_file dengan ebooklib mock.
    """
    mock_book = MagicMock()
    mock_item = MagicMock()
    mock_item.get_content.return_value = b"<html><body>Konten EPUB</body></html>"
    # ebooklib.ITEM_DOCUMENT is value 9
    mock_item.get_type.return_value = 9 
    mock_book.get_items_of_type.return_value = [mock_item]
    mock_read_epub.return_value = mock_book
    
    mock_soup = MagicMock()
    mock_soup.get_text.return_value = "Konten EPUB"
    mock_bs4.return_value = mock_soup

    result = load_epub_file("dummy.epub")
    assert result == "Konten EPUB"

@patch("os.path.exists")
@patch("ingestion.loader.load_text_file")
def test_load_document(mock_load_text: MagicMock, mock_exists: MagicMock) -> None:
    """
    Menguji fungsi penentu format load_document.
    """
    mock_exists.return_value = True
    mock_load_text.return_value = "Hasil text"
    
    result = load_document("file.txt")
    assert result == "Hasil text"
    mock_load_text.assert_called_once_with("file.txt")

    with pytest.raises(ValueError):
        load_document("file.invalid")
