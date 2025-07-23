import os
import textract
import pptx
import csv
import markdown
import docx
import PyPDF2

def parse_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return parse_pdf(file_path)
    elif extension == ".docx":
        return parse_docx(file_path)
    elif extension == ".pptx":
        return parse_pptx(file_path)
    elif extension == ".csv":
        return parse_csv(file_path)
    elif extension == ".txt":
        return parse_txt(file_path)
    elif extension == ".md":
        return parse_md(file_path)
    else:
        raise ValueError("Unsupported file format")

def parse_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return split_text(text)

def parse_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return split_text(text)

def parse_pptx(file_path):
    prs = pptx.Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return split_text(text)

def parse_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        text = "\n".join(["\t".join(row) for row in reader])
    return split_text(text)

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return split_text(text)

def parse_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
    return split_text(html)

def split_text(text, max_chunk_size=500):
    words = text.split()
    chunks = []
    current = []
    count = 0
    for word in words:
        current.append(word)
        count += len(word)
        if count >= max_chunk_size:
            chunks.append(" ".join(current))
            current = []
            count = 0
    if current:
        chunks.append(" ".join(current))
    return chunks

