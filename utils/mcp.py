import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

class MCPMessage:
    """
    Structured message format for agent communication.
    
    Recommended fields:
    - trace_id:    root request identifier (stays the same across one user request)
    - message_id:  unique id for *this* message
    - correlation_id: usually same as trace_id, but can be used to group related messages
    - timestamp:   when the message was created (UTC)
    - version:     schema version (helps when formats evolve)
    """
    def __init__(
        self,
        sender: str,
        receiver: str,
        msg_type: str,
        payload: Dict[str, Any],
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        version: str = "1.0",
    ):
        self.message_id = str(uuid.uuid4())
        self.trace_id = trace_id or str(uuid.uuid4())
        self.correlation_id = correlation_id or self.trace_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.sender = sender
        self.receiver = receiver
        self.type = msg_type
        self.version = version
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "version": self.version,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        msg = cls(
            sender=data["sender"],
            receiver=data["receiver"],
            msg_type=data["type"],
            payload=data["payload"],
            trace_id=data.get("trace_id"),
            correlation_id=data.get("correlation_id"),
            version=data.get("version", "1.0"),
        )
        # Preserve original message_id if present
        if "message_id" in data:
            msg.message_id = data["message_id"]
        if "timestamp" in data:
            msg.timestamp = data["timestamp"]
        return msg


# Convenience factory functions (drop-in replacement for your original helpers)

def generate_trace_id() -> str:
    return str(uuid.uuid4())


def create_message(
    sender: str,
    receiver: str,
    msg_type: str,
    payload: Dict[str, Any],
    trace_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    version: str = "1.0",
) -> Dict[str, Any]:
    """
    Compatibility wrapper — returns dict like your original code
    """
    msg = MCPMessage(
        sender=sender,
        receiver=receiver,
        msg_type=msg_type,
        payload=payload,
        trace_id=trace_id,
        correlation_id=correlation_id,
        version=version,
    )
    return msg.to_dict()

