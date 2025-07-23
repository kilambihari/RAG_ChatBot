import os
import streamlit as st
import uuid

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.file_handler import save_file
from utils.mcp import create_message, parse_message

st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")
st.title("📄 Upload a Document")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    help="Limit 200MB per file • PDF, DOCX, PPTX, CSV, TXT, MD"
)

if uploaded_file:
    save_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {save_path}")

    trace_id = str(uuid.uuid4())

    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()
    llm_agent = LLMResponseAgent()

    # --- MCP: Step 1 – Ingest document
    ingest_msg = create_message(
        sender="UI",
        receiver="IngestionAgent",
        msg_type="DOCUMENT_UPLOAD",
        trace_id=trace_id,
        payload={"file_path": save_path}
    )

    ingestion_response = ingestion_agent.handle_message(ingest_msg)
    chunks = ingestion_response["payload"]["chunks"]

    # Display sample chunks
    st.subheader("📄 Document Chunks")
    for i, chunk in enumerate(chunks[:5]):
        st.markdown(f"**Chunk {i+1}:** {chunk.page_content[:500]}{'...' if len(chunk.page_content) > 500 else ''}")

    # --- MCP: Step 2 – Store chunks in vector DB
    store_response = retrieval_agent.handle_message(ingestion_response)
    if store_response["payload"]["status"] == "stored":
        st.success("✅ Document stored in vector database.")

    # --- MCP: Step 3 – Q&A
    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question about the document")

    if question:
        query_msg = create_message(
            sender="UI",
            receiver="RetrievalAgent",
            msg_type="QUERY",
            trace_id=trace_id,
            payload={"query": question}
        )

        retrieval_response = retrieval_agent.handle_message(query_msg)
        llm_response = llm_agent.handle_message(retrieval_response)

        final_answer = llm_response["payload"]["answer"]
        st.markdown("### 🧠 Answer")
        st.markdown(final_answer)
