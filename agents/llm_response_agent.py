from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.mcp import create_message

class LLMResponseAgent:
    def __init__(self, agent_id="LLMResponseAgent", api_key=None):
        self.agent_id = agent_id
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    def handle_message(self, message: dict) -> dict:
        sender = message.get("from")
        receiver = message.get("to")
        msg_type = message.get("type")
        trace_id = message.get("trace_id", None)
        payload = message.get("payload", {})

        query = payload.get("query")
        context = payload.get("context")

        if not query or not context:
            raise ValueError("Missing query or context in payload")

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion:\n{question}"
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run({"context": context, "question": query})

        return create_message(
            sender=self.agent_id,
            receiver=sender,
            msg_type="LLM_RESPONSE",
            trace_id=trace_id,
            payload={"answer": answer}
        )

