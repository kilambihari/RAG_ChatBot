from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

class RetrievalAgent:
    def __init__(self, api_key, model="models/embedding-001"):
        self.embed_model = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )
        self.vector_store = None
        self.documents = []

    def build_index(self, chunks):
        # Wrap chunks as LangChain Documents
        self.documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Create FAISS vector store from documents
        self.vector_store = FAISS.from_documents(self.documents, self.embed_model)

    def retrieve(self, query, top_k=5):
        if not self.vector_store:
            return ["Index not built. Please upload and ingest a document first."]
        
        results = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

