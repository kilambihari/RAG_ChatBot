import os
import pickle
from sentence_transformers import SentenceTransformer
from utils.parser import parse_document
from utils.mcp import MCPMessage

class IngestionAgent:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def handle_message(self, message: MCPMessage) -> MCPMessage:
        file_path = message.content
        print(f"📄 [IngestionAgent] Received file path: {file_path}")

        chunks = parse_document(file_path)
        print(f"🧩 [IngestionAgent] Parsed {len(chunks)} chunks from document")

        embeddings = self.model.encode(chunks, convert_to_tensor=False)
        print(f"🔢 [IngestionAgent] Generated embeddings for chunks")

        # Save vectorstore to file
        vectorstore = {
            "chunks": chunks,
            "embeddings": [e.tolist() for e in embeddings],
        }

        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

        print("✅ [IngestionAgent] Saved vectorstore as vectorstore.pkl")

        return MCPMessage(
            sender="IngestionAgent",
            receiver="App",
            type="result",
            content="Document parsed and embeddings saved!"
        )
