# 🧠 Agentic RAG Chatbot (Gemini + MCP)

A multi-agent **Retrieval-Augmented Generation (RAG)** chatbot powered by **Gemini (pro/flash)** and **Model Context Protocol (MCP)**. It allows users to upload documents, intelligently parse them, retrieve relevant content, and generate answers using an LLM—all with agentic communication.

## 🚀 Features

- 📄 Upload and parse documents (PDF, DOCX, TXT, PPTX, CSV, MD)
- 🤖 Agent-based architecture using Model Context Protocol
- 🔎 Embedding + vector store search (FAISS + SentenceTransformers)
- 💬 Natural language Q&A with Gemini Pro or Flash
- 🧩 Modular Agents: IngestionAgent, RetrievalAgent, LLMResponseAgent
- 🌐 Deployable directly via Streamlit Cloud
- ✅ Supports Gemini API with `gemini-pro` or `gemini-1.5-flash`

---

## 🧱 Architecture

```bash
📁 rag_chatbot/
├── app.py
├── agents/
│   ├── ingestion_agent.py
│   ├── retrieval_agent.py
│   └── llm_response_agent.py
├── utils/
│   ├── parser.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── mcp.py
├── requirements.txt
└── README.md


## 🚀 Live Demo

🔗 **Streamlit App**: [Click here to view the RAG Chatbot](https://ragchatbot-peybjfbeevhbbhx4x4dmqo.streamlit.app/)
