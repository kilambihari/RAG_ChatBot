import uuid

def create_message(sender, receiver, msg_type, trace_id=None, payload=None):
    """
    Creates a standardized MCP message dictionary.
    """
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id or str(uuid.uuid4()),
        "payload": payload or {},
    }

def parse_message(message):
    """
    Unpacks a message dictionary into its components.
    """
    return (
        message.get("sender"),
        message.get("receiver"),
        message.get("type"),
        message.get("trace_id"),
        message.get("payload"),
    )

