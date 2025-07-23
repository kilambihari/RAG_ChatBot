# utils/parser.py

import os
import fitz  # PyMuPDF
import csv
import docx
import pptx
import markdown
import pandas as pd

def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".pptx":
        return parse_pptx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".md":
        return parse_md(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def parse_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_pptx(file_path):
    prs = pptx.Presentation(file_path)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def parse_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return markdown.markdown(f.read())
