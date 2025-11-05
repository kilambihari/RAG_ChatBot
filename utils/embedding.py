import google.generativeai as genai
import streamlit as st

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

EMBED_MODEL = "models/embedding-001"

def get_gemini_embedding(chunks):
    if not chunks or len(chunks) == 0:
        st.warning("⚠️ No text chunks provided for embedding.")
        return []

    embeddings = []
    for i, text in enumerate(chunks):
        try:
            response = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
                title=f"Chunk-{i+1}"
            )

            # Safely extract embedding
            embedding = response.get("embedding")
            if embedding and len(embedding) > 0:
                embeddings.append(embedding)
            else:
                st.warning(f"⚠️ Empty embedding returned for chunk {i+1}.")

        except Exception as e:
            st.warning(f"❌ Embedding failed for chunk {i+1}: {e}")

    st.info(f"✅ Generated {len(embeddings)} embeddings out of {len(chunks)} chunks.")
    return embeddings


def query_gemini_llm(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API failed: {e}")
        return ""

