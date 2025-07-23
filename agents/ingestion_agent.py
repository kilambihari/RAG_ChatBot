import os
from utils.parser import parse_file
from utils.chunker import chunk_text
from utils.mcp import create_message

class IngestionAgent:
    def __init__(self, agent_id="IngestionAgent"):
        self.agent_id = agent_id

    def handle_message(self, message: dict) -> dict:
        # ✅ Safely extract values from message
        sender = message.get("from")
        receiver = message.get("to")
        msg_type = message.get("type")
        trace_id = message.get("trace_id", None)
        payload = message.get("payload", {})

        file_path = payload.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        # ✅ Step 1: Parse document into raw text
        text = parse_file(file_path)

        # ✅ Step 2: Split text into chunks
        chunks = chunk_text(text)

        # ✅ Step 3: Package response via MCP format
        response = create_message(
            sender=self.agent_id,
            receiver=sender,
            msg_type="INGESTION_RESPONSE",
            trace_id=trace_id,
            payload={"chunks": chunks}
        )
        return response
