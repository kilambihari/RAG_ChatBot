import os
from PyPDF2 import PdfReader
import docx
import pptx
import csv

def parse_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return [page.extract_text() for page in reader.pages if page.extract_text()]

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return [para.text for para in doc.paragraphs if para.text.strip()]

    elif ext == ".pptx":
        prs = pptx.Presentation(file_path)
        slides = []
        for slide in prs.slides:
            text = " ".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
            if text.strip():
                slides.append(text)
        return slides

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()

    elif ext == ".csv":
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        return rows

    elif ext == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()

    else:
        raise ValueError(f"Unsupported file format: {ext}")
