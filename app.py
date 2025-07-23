import os
import uuid
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.mcp import create_message
from utils.file_handler import save_file

# --- Streamlit Page Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")
st.title("📄 Upload a Document")
st.markdown("Upload a document to enable question answering:")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "pptx", "csv", "txt", "md"])
if uploaded_file:
    file_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {file_path}")

    # --- Initialize Agents ---
    trace_id = str(uuid.uuid4())
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()
    llm_agent = LLMResponseAgent()

    # --- Ingest File ---
    ingestion_message = create_message("App", "IngestionAgent", "INGEST", trace_id, {"file_path": file_path})
    ingested_data = ingestion_agent.handle_message(ingestion_message)

    # --- Store for Retrieval ---
    retrieval_message = create_message("App", "RetrievalAgent", "STORE", trace_id, {
        "content": ingested_data["payload"]["content"],
        "doc_id": file_path
    })
    retrieval_response = retrieval_agent.handle_message(retrieval_message)
    st.success("📚 Document processed and stored!")

    # --- Ask a Question ---
    query = st.text_input("Ask a question based on the uploaded document:")
    if query:
        query_message = create_message("App", "RetrievalAgent", "RETRIEVE", trace_id, {
            "query": query,
            "top_k": 3
        })
        retrieved_chunks = retrieval_agent.handle_message(query_message)

        llm_message = create_message("App", "LLMResponseAgent", "GENERATE", trace_id, {
            "query": query,
            "chunks": retrieved_chunks["payload"]["chunks"]
        })
        final_response = llm_agent.handle_message(llm_message)
        st.markdown(f"🧠 **Answer:** {final_response['payload']['response']}")


               
