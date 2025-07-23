import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.file_handler import save_file
from utils.mcp import create_message

# --- Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")

st.title("📄 Upload a Document")
st.markdown("Upload a document")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"]
)

if uploaded_file:
    file_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {file_path}")

    # --- Ingestion ---
    ingestion_agent = IngestionAgent()
    message = create_message(
        sender="App",
        receiver="IngestionAgent",
        msg_type="FILE_UPLOAD",
        trace_id="1",
        payload={"file_path": file_path}
    )
    ingestion_response = ingestion_agent.handle_message(message)

    # --- Display Chunks ---
    st.subheader("📄 Document Chunks")
    for i, chunk in enumerate(ingestion_response["payload"]["chunks"]):
        st.markdown(f"**Chunk {i+1}:** {chunk[:500]}...")

    # --- Retrieval ---
    retrieval_agent = RetrievalAgent()
    retrieval_response = retrieval_agent.handle_message(ingestion_response)

    # --- User Query ---
    st.subheader("❓ Ask a Question")
    query = st.text_input("Enter your question:")
    if query:
        context_chunks = retrieval_agent.retrieve(query, k=3)
        context = "\n\n".join(context_chunks)

        llm_agent = LLMResponseAgent(api_key=st.secrets["GEMINI_API_KEY"])
        llm_message = create_message(
            sender="App",
            receiver="LLMResponseAgent",
            msg_type="USER_QUERY",
            trace_id="2",
            payload={"query": query, "context": context}
        )
        llm_response = llm_agent.handle_message(llm_message)

        st.subheader("🧠 Answer")
        st.write(llm_response["payload"]["answer"])


               
