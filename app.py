import os
import streamlit as st
from utils.file_handler import save_file
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent

# --- Set Page Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")

# --- Load API Keys from Streamlit Secrets ---
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# --- App Title ---
st.title("📄 Upload a Document")
st.write("Upload a document")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a document", type=["pdf", "docx", "pptx", "csv", "txt", "md"]
)

if uploaded_file:
    # --- Save uploaded file ---
    save_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {save_path}")

    # --- Chunk document using IngestionAgent ---
    ingestion_agent = IngestionAgent()
    chunks = ingestion_agent.run(save_path)

    st.subheader("📄 Document Chunks")
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk[:500]}{'...' if len(chunk) > 500 else ''}")

    # --- Store Chunks into Vector Store using RetrievalAgent ---
    retrieval_agent = RetrievalAgent()
    retrieval_agent.store(chunks)
    st.success("✅ Chunks stored in vector index.")

    # --- Ask Questions Section ---
    st.subheader("💬 Ask a Question from Document")
    user_question = st.text_input("Enter your question:")

    if user_question:
        relevant_chunks = retrieval_agent.retrieve(user_question)

        st.subheader("🔍 Retrieved Chunks")
        if not relevant_chunks:
            st.warning("⚠️ No relevant chunks found for this question.")
        else:
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk[:500]}{'...' if len(chunk) > 500 else ''}")

            # --- Generate response from LLMResponseAgent ---
            llm_agent = LLMResponseAgent()
            answer = llm_agent.run(relevant_chunks, user_question)

            st.subheader("🧠 LLM Response")
            st.markdown(answer)


