import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

EMBED_MODEL = "models/embedding-001"

def get_gemini_embedding(chunks):
    embeddings = []
    st.write(f"🔍 Received {len(chunks)} chunks for embedding")

    for i, text in enumerate(chunks):
        try:
            if not text.strip():
                st.warning(f"⚠️ Skipping empty chunk #{i}")
                continue

            st.write(f"🟢 Sending chunk #{i} — length {len(text)}")
            response = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
                title="Doc Chunk"
            )

            emb = response.get("embedding", None)
            if emb:
                embeddings.append(emb)
                st.write(f"✅ Got embedding length: {len(emb)}")
            else:
                st.error(f"❌ No embedding returned for chunk #{i}")

        except Exception as e:
            st.error(f"❌ Embedding failed for chunk #{i}: {e}")

    st.write(f"📦 Total embeddings generated: {len(embeddings)}")
    return embeddings

