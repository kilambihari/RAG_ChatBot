# agents/ingestion_agent.py

from utils.parser import parse_file
from utils.mcp import parse_message, create_message

class IngestionAgent:
    def __init__(self, name="IngestionAgent"):
        self.name = name

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type == "INGEST":
            file_path = payload["file_path"]
            content = parse_file(file_path)
            return create_message(
                self.name,
                sender,
                "INGESTED",
                trace_id,
                {"content": content}
            )
        else:
            return create_message(
                self.name,
                sender,
                "ERROR",
                trace_id,
                {"error": f"Unsupported message type: {msg_type}"}
            )
