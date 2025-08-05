import os
import PyPDF2
import pandas as pd
import docx
import pptx
import markdown

def parse_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(page.extract_text() for page in reader.pages)
    return [text[i:i+1000] for i in range(0, len(text), 1000)]

def parse_docx(file_path):
    doc = docx.Document(file_path)
    return [p.text for p in doc.paragraphs if p.text.strip()]

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def parse_csv(file_path):
    df = pd.read_csv(file_path)
    return [row for row in df.astype(str).apply(lambda x: " | ".join(x), axis=1)]

def parse_pptx(file_path):
    prs = pptx.Presentation(file_path)
    text_runs = []
    for slide in prs.slides:
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text.strip()
                if text:
                    slide_text.append(text)
        if slide_text:
            text_runs.append(" ".join(slide_text))
    return text_runs

def parse_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def parse_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".pptx":
        return parse_pptx(file_path)
    elif ext == ".md":
        return parse_md(file_path)
    else:
        raise ValueError("Unsupported file format")

