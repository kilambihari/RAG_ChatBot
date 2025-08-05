# 🧠 Agentic RAG Chatbot (Gemini + MCP)

A multi-agent **Retrieval-Augmented Generation (RAG)** chatbot powered by **Gemini (pro/flash)** and **Model Context Protocol (MCP)**. It allows users to upload documents, intelligently parse them, retrieve relevant content, and generate answers using an LLM—all with agentic communication.

## 🚀 Features

📄 Multi-format Document Upload: Supports PDF, DOCX, TXT, PPTX, CSV, MD for ingestion

🤖 Agent-based Architecture: Built with Model Context Protocol (MCP) to facilitate modular communication

🔎 Semantic Search: Uses Gemini Embeddings + FAISS vector store for fast, relevant retrieval

💬 LLM-Powered Chatbot: Natural language Q&A via Gemini Pro or Gemini 1.5 Flash

🧩 Modular Agents:

IngestionAgent: Parses and chunks documents

RetrievalAgent: Retrieves relevant content using vector similarity

LLMResponseAgent: Generates human-like answers using Gemini LLMs

🌐 Deployable via Streamlit Cloud: Fully web-based, no local setup required

✅ Gemini API Ready: Easily switch between gemini-pro or gemini-1.5-flash

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
