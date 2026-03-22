import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
import os

class DocumentProcessor:
    def __init__(self, chunk_size=400, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file using PyMuPDF."""
        text = ""
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text

    def clean_text(self, text: str) -> str:
        """Cleans extracted text - removes double spaces, newlines, etc."""
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """Splits text into chunks with metadata."""
        words = text.split(' ')
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Simple metadata: document name, chunk index, word count
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc_name,
                    "chunk_id": chunk_idx,
                    "word_count": len(chunk_words)
                }
            })
            
            start += (self.chunk_size - self.chunk_overlap)
            chunk_idx += 1
            
        return chunks

    def process_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Complete workflow: extract, clean, chunk."""
        doc_name = os.path.basename(pdf_path)
        raw_text = self.extract_text_from_pdf(pdf_path)
        clean_text = self.clean_text(raw_text)
        return self.chunk_text(clean_text, doc_name)
