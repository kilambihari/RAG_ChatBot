import os
import pickle
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import streamlit as st  # ✅ to access secrets

class RetrievalAgent:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # ✅ Use secrets from Streamlit
        hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        gemini_key = st.secrets["GEMINI_API_KEY"]

        self.embeddings = HuggingFaceHubEmbeddings(
            repo_id=model_name,
            huggingfacehub_api_token=hf_token
        )

        self.vector_store = None

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
            google_api_key=gemini_key
        )

        self.rewrite_chain = self._build_rewriter_chain()

    def _build_rewriter_chain(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an assistant helping improve search in a resume-based question-answering system.
Rewrite the question into a form that matches how information is typically written in resumes.

Original question: {question}
Improved search query:"""
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def load_vector_store(self, index_path: str):
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.vector_store = pickle.load(f)

    def save_vector_store(self, index_path: str):
        if self.vector_store:
            with open(index_path, "wb") as f:
                pickle.dump(self.vector_store, f)

    def set_vector_store(self, docs):
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return ["⚠️ No documents have been stored yet."]

        try:
            rewritten = self.rewrite_chain.run(query)
        except Exception as e:
            rewritten = query
            print(f"[⚠️ Rewrite failed] {e}")

        print(f"[🔁 Rewritten Query] {rewritten}")
        docs = self.vector_store.similarity_search(rewritten, k=k)
        return [doc.page_content for doc in docs]


