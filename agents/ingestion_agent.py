import os
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, TextLoader,
    CSVLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.mcp import create_message

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

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = message.values()
        file_path = payload["file_path"]

        loader = self._get_loader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        return create_message(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            msg_type="DOCUMENT_CHUNKS",
            trace_id=trace_id,
            payload={"chunks": chunks}
        )
