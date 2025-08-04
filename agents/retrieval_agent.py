from utils.vector_store import search_similar_chunks

class RetrievalAgent:
    def __init__(self):
        pass

    def handle_message(self, message):
        query = message["payload"]["query"]
        doc_id = message["payload"]["doc_id"]
        top_chunks = search_similar_chunks(doc_id, query)
        return {"top_chunks": top_chunks, "query": query}
