import uuid

def generate_trace_id():
    return str(uuid.uuid4())

def create_message(sender, receiver, msg_type, trace_id, payload):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id,
        "payload": payload
    }


