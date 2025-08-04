import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.mcp import create_message, generate_trace_id

os.makedirs("data", exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

st.set_page_config(page_title="Agentic RAG with MCP", layout="wide")
st.title("Agentic RAG Chatbot (with MCP)")

uploaded = st.file_uploader("Upload a document", type=["pdf","docx","txt","md","csv","pptx"])
if uploaded:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", uploaded.name)
    with open(path, "wb") as f: 
        f.write(uploaded.read())
    st.success(f"Uploaded {uploaded.name}")

ingestor = IngestionAgent()
retriever = RetrievalAgent()
responder = LLMResponseAgent()

if uploaded and st.button("Process Document"):
    trace = generate_trace_id()
    msg_ingest = create_message("User","IngestionAgent","INGEST_REQUEST",trace,{"file_path":path})
    ingest_resp = ingestor.handle_message(msg_ingest)
    st.success("Document parsed and indexed.")

    user_q = st.text_input("Enter your question:")
    if user_q:
        msg_retrieve = create_message("User","RetrievalAgent","RETRIEVE_REQUEST",trace,{"query":user_q,"doc_id":ingest_resp["doc_id"]})
        retrieve_resp = retriever.handle_message(msg_retrieve)

        msg_llm = create_message("RetrievalAgent","LLMResponseAgent","CONTEXT_RESPONSE",trace,retrieve_resp)
        llm_resp = responder.handle_message(msg_llm)

        st.markdown("### 🤖 Answer")
        st.write(llm_resp["answer"])
        st.markdown("### 📚 Source Context")
        for c in llm_resp["source_chunks"]:
            st.code(c)
