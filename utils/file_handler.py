import os

UPLOAD_DIR = "uploads"

def save_file(uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def chunk_text(text, chunk_size=500):
    """
    Splits a long string into smaller chunks of specified size.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
