import os
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.mcp import create_message, generate_trace_id

# ────────────────────────────────────────────────
#  Folders
# ────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

# ────────────────────────────────────────────────
#  Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG with MCP",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Agentic RAG Chatbot (with MCP)")

# ────────────────────────────────────────────────
#  Initialize session state
# ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

if "trace" not in st.session_state:
    st.session_state.trace = None

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

# ────────────────────────────────────────────────
#  Sidebar – Document status
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Status")
    if st.session_state.doc_id:
        st.success(f"Active document ID: {st.session_state.doc_id[:8]}…")
        if st.button("Clear current document", use_container_width=True):
            st.session_state.doc_id = None
            st.session_state.trace = None
            st.session_state.ingestion_done = False
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No document processed yet")

# ────────────────────────────────────────────────
#  File uploader
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf", "docx", "txt", "md", "csv", "pptx"],
    help="Supported formats: PDF, Word, Text, Markdown, CSV, PowerPoint"
)

if uploaded_file is not None:
    # ── Save file ─────────────────────────────────
    save_path = os.path.join("data", uploaded_file.name)

    # Only re-save if filename changed (avoid unnecessary writes)
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.success(f"Saved: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    else:
        st.info(f"Using already saved file: **{uploaded_file.name}**")

    # ── Process button ─────────────────────────────
    if not st.session_state.ingestion_done or st.session_state.last_uploaded_filename != uploaded_file.name:
        if st.button("📄 Process & Index Document", type="primary", use_container_width=True):
            with st.spinner("Ingesting document... (may take a while for large files)"):
                try:
                    ingestor = IngestionAgent()
                    trace = generate_trace_id()

                    msg = create_message(
                        sender="User",
                        msg_type="INGEST_REQUEST",
                        trace_id=trace,
                        payload={"file_path": save_path}
                    )

                    response = ingestor.handle_message(msg)

                    if "doc_id" in response and response["doc_id"]:
                        st.session_state.doc_id = response["doc_id"]
                        st.session_state.trace = trace
                        st.session_state.ingestion_done = True
                        st.session_state.last_uploaded_filename = uploaded_file.name
                        st.success(f"Document indexed successfully\n**doc_id:** {response['doc_id']}")
                    else:
                        st.error("Ingestion did not return a valid doc_id")

                except Exception as e:
                    st.error(f"Ingestion failed\n{str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

# ────────────────────────────────────────────────
#  Chat interface – only shown when we have a doc
# ────────────────────────────────────────────────
if st.session_state.doc_id:

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document → Generating answer…"):
                try:
                    retriever = RetrievalAgent()
                    responder = LLMResponseAgent()

                    # 1. Retrieve
                    retrieve_msg = create_message(
                        sender="User",
                        msg_type="RETRIEVE_REQUEST",
                        trace_id=st.session_state.trace,
                        payload={
                            "query": prompt,
                            "doc_id": st.session_state.doc_id
                        }
                    )

                    retrieve_result = retriever.handle_message(retrieve_msg)

                    # 2. Generate answer
                    llm_msg = create_message(
                        sender="RetrievalAgent",
                        msg_type="CONTEXT_RESPONSE",
                        trace_id=st.session_state.trace,
                        payload=retrieve_result
                    )

                    final_result = responder.handle_message(llm_msg)

                    answer = final_result.get("answer", "No answer generated.")
                    sources = final_result.get("source_chunks", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Source chunks", expanded=False):
                            for i, chunk in enumerate(sources, 1):
                                st.markdown(f"**Chunk {i}**")
                                st.code(chunk.strip(), language=None)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error during retrieval / generation\n{str(e)}")
                    st.exception(e)

else:
    # No document yet → helper message
    if st.session_state.last_uploaded_filename:
        st.info("Please click **Process & Index Document** to make the file searchable.")
    else:
        st.info("Upload a document to start asking questions about it.")
