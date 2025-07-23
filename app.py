# app.py

import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.file_handler import save_file

# --- Page Config ---
st.set_page_config(page_title="📄 RAG Chatbot", layout="centered")

# --- Instantiate Agents ---
ingestion_agent = IngestionAgent()
retrieval_agent = RetrievalAgent()
llm_agent = LLMResponseAgent()

# --- Upload UI ---
st.title("📄 Upload a Document")
st.markdown("Upload a document")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"]
)

if uploaded_file:
    file_path = save_file(uploaded_file)
    st.success(f"✅ File saved at: {file_path}")

    # IngestionAgent handles the file
    message = {
        "type": "file_uploaded",
        "from": "app",
        "to": "IngestionAgent",
        "payload": {"file_path": file_path}
    }
    ingestion_response = ingestion_agent.handle_message(message)

    # Display Chunks
    if ingestion_response["type"] == "document_chunks":
        st.subheader("📄 Document Chunks")
        for i, chunk in enumerate(ingestion_response["payload"]["chunks"]):
            st.markdown(f"**Chunk {i+1}:** {chunk}")

        # RetrievalAgent stores the chunks in vectorstore
        retrieval_response = retrieval_agent.handle_message(ingestion_response)

        if retrieval_response["type"] == "vectorstore_ready":
            st.success("✅ Vector store is ready!")

            # Input box for user query
            st.subheader("💬 Ask a Question")
            user_query = st.text_input("Enter your question")

            if user_query:
                query_message = {
                    "type": "query",
                    "from": "app",
                    "to": "RetrievalAgent",
                    "payload": {"query": user_query}
                }

                results = retrieval_agent.query(user_query)
                result_chunks = [doc.page_content for doc in results]

                # Send to LLM agent
                llm_message = {
                    "type": "query_and_context",
                    "from": "app",
                    "to": "LLMResponseAgent",
                    "payload": {
                        "query": user_query,
                        "chunks": result_chunks
                    }
                }

                llm_response = llm_agent.handle_message(llm_message)

                if llm_response["type"] == "llm_response":
                    st.subheader("🧠 Answer")
                    st.markdown(llm_response["payload"]["response"])

