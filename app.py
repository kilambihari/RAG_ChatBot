import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from utils.file_handler import save_file

# --- Page Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")

st.title("📄 Upload a Document")
st.markdown("Upload a document")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    help="Limit 200MB per file • PDF, DOCX, PPTX, CSV, TXT, MD"
)

if uploaded_file:
    save_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {save_path}")

    # Initialize Agents
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()

    # Ingest document
    chunks = ingestion_agent.run(save_path)
    if chunks:
        st.subheader("📄 Document Chunks")
        for i, chunk in enumerate(chunks):
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            st.markdown(f"**Chunk {i+1}:** {text[:500]}{'...' if len(text) > 500 else ''}")
        
        # Store chunks in vector DB
        store_message = {"action": "store", "data": chunks}
        store_response = retrieval_agent.handle_message(store_message)
        st.success("✅ Document stored in vector database")

        # Q&A section
        st.subheader("💬 Ask a Question")
        question = st.text_input("Enter your question about the document")

        if question:
            query_message = {"action": "query", "data": question}
            response = retrieval_agent.handle_message(query_message)

            st.markdown("### 🧠 Answer")
            st.markdown(response)

