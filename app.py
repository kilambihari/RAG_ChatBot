import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.file_handler import save_file
from utils.mcp import Message

# --- Page Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")
st.title("📄 Upload a Document")
st.markdown("Upload a document to enable question answering:")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    help="Supported formats: PDF, DOCX, PPTX, CSV, TXT, MD",
)

# --- Agent Initialization ---
ingestion_agent = IngestionAgent()
retrieval_agent = RetrievalAgent()
llm_response_agent = LLMResponseAgent()

# --- File Processing & Embedding ---
if uploaded_file is not None:
    file_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {file_path}")

    # Create and send message to Ingestion Agent
    ingestion_message = Message(
        sender="app",
        receiver="ingestion_agent",
        content={"file_path": file_path},
        metadata={"purpose": "ingest_document"},
    )
    try:
        ingested_data = ingestion_agent.handle_message(ingestion_message)
        st.success("✅ Document successfully ingested and embedded.")
    except Exception as e:
        st.error(f"❌ Error in ingestion: {e}")

# --- Chatbot UI ---
st.markdown("---")
st.header("💬 Ask a Question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Type your question here")

if st.button("Ask") and user_query:
    # Add user query to chat history
    st.session_state.chat_history.append(("user", user_query))

    # Message to RetrievalAgent
    retrieval_message = Message(
        sender="app",
        receiver="retrieval_agent",
        content={"query": user_query},
        metadata={"purpose": "retrieve_relevant_chunks"},
    )
    try:
        retrieved_chunks = retrieval_agent.handle_message(retrieval_message)

        # Message to LLMResponseAgent
        llm_message = Message(
            sender="app",
            receiver="llm_response_agent",
            content={"query": user_query, "context": retrieved_chunks},
            metadata={"purpose": "generate_response"},
        )
        final_answer = llm_response_agent.handle_message(llm_message)
        st.session_state.chat_history.append(("bot", final_answer))

    except Exception as e:
        st.error(f"❌ Error during retrieval or LLM response: {e}")

# --- Display Chat History ---
if st.session_state.chat_history:
    st.markdown("### 🧠 Chat History")
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**👤 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")
