import os
import streamlit as st
import uuid

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.file_handler import save_file
from utils.mcp import create_message, parse_message

# --- Page Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")

st.title("📄 Upload a Document")
st.markdown("This chatbot supports **PDF, DOCX, PPTX, CSV, TXT, and MD** formats. Max size: 200MB.")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    help="Limit 200MB per file • PDF, DOCX, PPTX, CSV, TXT, MD"
)

if uploaded_file:
    # --- Save File ---
    save_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: `{save_path}`")

    # --- Initialize Agents ---
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()
    llm_agent = LLMResponseAgent()

    # --- Ingest and Chunk ---
    chunks = ingestion_agent.run(save_path)
    if chunks:
        st.subheader("📄 Document Chunks")
        for i, chunk in enumerate(chunks[:10]):  # show only first 10 chunks
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            st.markdown(f"**Chunk {i+1}:** {text[:500]}{'...' if len(text) > 500 else ''}")

        # --- Store in VectorDB ---
        store_message = {"action": "store", "data": chunks}
        retrieval_agent.handle_message(store_message)
        st.success("✅ Document stored in vector database.")

        # --- Question Answering ---
        st.subheader("💬 Ask a Question")
        question = st.text_input("Enter your question about the document")

        if question:
            trace_id = str(uuid.uuid4())
            query_msg = create_message(
                sender="User",
                receiver="RetrievalAgent",
                msg_type="query",
                trace_id=trace_id,
                payload=question
            )

            # --- Get relevant chunks ---
            raw_response = retrieval_agent.handle_message(query_msg)
            relevant_chunks = raw_response.split("\n\n")

            # --- Generate Answer ---
            final_answer = llm_agent.generate_response(question, relevant_chunks)

            st.markdown("### 🧠 Answer")
            st.markdown(final_answer)

