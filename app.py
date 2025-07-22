import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from utils.mcp import create_message

# --- Streamlit Setup ---
st.set_page_config(page_title="RAG Chatbot - Upload & Ingest", layout="centered")
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

    st.success(f"✅ File saved at: `{file_path}`")

    # --- Ingest Using Agent ---
    ingestion_agent = IngestionAgent()
    message = create_message("App", "IngestionAgent", "ingest", "123", {"file_path": file_path})
    response = ingestion_agent.handle_message(message)

    # --- Show Result ---
    chunks = response["payload"].get("chunks", [])
    st.subheader("📄 Document Chunks")
    for i, chunk in enumerate(chunks[:5]):
        st.markdown(f"**Chunk {i+1}:** {chunk}")

    # --- Initialize Retrieval Agent with API Key ---
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Gemini API key not found. Set GEMINI_API_KEY in environment or Streamlit secrets.")
    else:
        retrieval_agent = RetrievalAgent()

        # --- Store Chunks ---
        store_message = create_message("App", "RetrievalAgent", "store", "456", {"chunks": chunks})
        store_response = retrieval_agent.handle_message(store_message)

        st.success("✅ Chunks stored in vector index.")

        # --- Ask a Question ---
        st.subheader("💬 Ask a Question from Document")
        query = st.text_input("Enter your question:")

        if query:
            query_message = create_message("App", "RetrievalAgent", "retrieve", "789", {"query": query})
            query_response = retrieval_agent.handle_message(query_message)
            matches = query_response["payload"].get("matches", [])

            st.subheader("🔍 Retrieved Chunks")
            if matches:
                for i, match in enumerate(matches):
                    text = match.get("text", match)  # fallback if not dict
                    st.markdown(f"**Match {i+1}:** {text}")
            else:
                st.warning("No relevant chunks found for this question.")





