import os
from utils.parsers.pdf import parse_pdf
from utils.parsers.docx import parse_docx
from utils.parsers.txt import parse_txt
from utils.parsers.pptx import parse_pptx
from utils.parsers.csv import parse_csv
from utils.parsers.md import parse_md

def parse_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".pptx":
        return parse_pptx(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".md":
        return parse_md(file_path)
    else:
        raise ValueError("Unsupported file format")
