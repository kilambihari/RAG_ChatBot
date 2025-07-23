import os
import tempfile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

class RetrievalAgent:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None

    def _create_vectorstore(self, chunks):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    def _create_qa_chain(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.3
        )
        retriever = self.vectorstore.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

    def handle_message(self, message):
        action = message.get("action")
        data = message.get("data")

        if action == "store":
            self._create_vectorstore(data)
            self._create_qa_chain()
            return "✅ Vectorstore created and LLM QA chain initialized."
        
        elif action == "query":
            if self.qa_chain is None:
                return "❌ QA chain is not initialized. Please upload a document first."
            return self.qa_chain.run(data)

        else:
            return f"❌ Unknown action: {action}"

