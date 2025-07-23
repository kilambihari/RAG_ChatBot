import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.file_handler import save_file
from utils.mcp import create_message

# --- App Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")
st.title("📄 Upload a Document")
st.markdown("Upload a document to enable question answering:")

# --- Agent Setup ---
ingestion_agent = IngestionAgent()
retrieval_agent = RetrievalAgent()
llm_agent = LLMResponseAgent()

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    help="Limit 200MB per file • PDF, DOCX, PPTX, CSV, TXT, MD"
)

if uploaded_file:
    file_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {file_path}")

    # --- Ingest Document ---
    trace_id = "session-001"
    ingestion_message = create_message(
        sender="App",
        receiver="IngestionAgent",
        msg_type="INGEST",
        trace_id=trace_id,
        payload={"file_path": file_path}
    )

    ingestion_result = ingestion_agent.handle_message(ingestion_message)

    if ingestion_result["type"] == "READY":
        st.success("✅ Document ingestion complete. You can now ask a question.")

        # --- Question Input ---
        user_query = st.text_input("Ask a question about the document:")

        if user_query:
            # --- Send user query to LLM Agent ---
            query_msg = create_message(
                sender="App",
                receiver="LLMResponseAgent",
                msg_type="USER_QUERY",
                trace_id=trace_id,
                payload={"query": user_query}
            )
            query_forward = llm_agent.handle_message(query_msg)

            # --- Pass query embedding to RetrievalAgent ---
            retrieved = retrieval_agent.handle_message(query_forward)

            # --- Send retrieved chunks back to LLM Agent ---
            final_answer_msg = llm_agent.handle_message(retrieved)

            # --- Display Answer ---
            final_answer = final_answer_msg["payload"]["response"]
            st.markdown("### 📌 Answer:")
            st.success(final_answer)
