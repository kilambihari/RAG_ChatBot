import os

def save_file(uploaded_file, upload_dir="uploads"):
    """Save uploaded file to disk and return its path."""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    save_path = os.path.join(upload_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path

