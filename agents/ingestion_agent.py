# agents/ingestion_agent.py

import fitz  # PyMuPDF
import docx
import pandas as pd
import os
from pptx import Presentation

def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext == "pptx":
        return extract_text_from_pptx(file_path)
    elif ext == "csv":
        return extract_text_from_csv(file_path)
    elif ext in ["txt", "md"]:
        return extract_text_from_txt(file_path)
    else:
        return ""

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
