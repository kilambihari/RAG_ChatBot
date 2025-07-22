
import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from utils.mcp import create_message

# --- Streamlit Setup ---
st.set_page_config(page_title=" RAG Chatbot - Upload & Ingest", layout="centered")
st.title("📄 Upload a Document")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "pptx", "csv", "txt", "md"])

# --- Handle File Upload ---
if uploaded_file:
    # Save file to disk
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved at: {file_path}")

    # --- Ingest Using Agent ---
    ingestion_agent = IngestionAgent()
    message = create_message("App", "IngestionAgent", "ingest", "123", {"file_path": file_path})
    response = ingestion_agent.handle_message(message)

    # --- Show Result ---
    chunks = response["payload"].get("chunks", [])
    st.subheader(" Document Chunks")
    for i, chunk in enumerate(chunks[:5]):  # show only first 5 chunks
        st.markdown(f"**Chunk {i+1}:** {chunk}")

from agents.retrieval_agent import RetrievalAgent
from utils.mcp import create_message  # already imported above

# --- Initialize Retrieval Agent ---
retrieval_agent = RetrievalAgent()

# --- Store Chunks in Retrieval Agent ---
store_message = create_message("App", "RetrievalAgent", "store", "456", {"chunks": chunks})
store_response = retrieval_agent.handle_message(store_message)

st.success("Chunks stored in retrieval index.")


