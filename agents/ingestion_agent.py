from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, TextLoader,
    CSVLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class IngestionAgent:
    def _get_loader(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return PyMuPDFLoader(file_path)
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(file_path)
        elif ext == ".pptx":
            return UnstructuredPowerPointLoader(file_path)
        elif ext == ".txt":
            return TextLoader(file_path)
        elif ext == ".csv":
            return CSVLoader(file_path)
        elif ext == ".md":
            return UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def run(self, file_path):
        loader = self._get_loader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        return chunks


