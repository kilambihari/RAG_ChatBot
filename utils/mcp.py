

def create_message(sender, receiver, msg_type, trace_id, payload):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id,
        "payload": payload,
    }

def parse_message(message):
    return (
        message["sender"],
        message["receiver"],
        message["type"],
        message["trace_id"],
        message["payload"]
    )

