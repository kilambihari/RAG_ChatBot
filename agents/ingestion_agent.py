from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class IngestionAgent:
    def __init__(self):
        self.supported_loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pptx": UnstructuredPowerPointLoader,
        }

    def _get_loader(self, file_path):
        for ext, loader_class in self.supported_loaders.items():
            if file_path.endswith(ext):
                return loader_class(file_path)
        raise ValueError(f"Unsupported file format: {file_path}")

    def run(self, file_path):
        loader = self._get_loader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks


