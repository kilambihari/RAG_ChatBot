import google.generativeai as genai
import numpy as np
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def get_gemini_embedding(chunks):
    model = genai.GenerativeModel("gemini-1.5-flash")
    embeddings = []

    for i, text in enumerate(chunks):
        try:
            prompt = f"Convert the following text into a 768-dimensional numerical embedding vector:\n{text}"
            response = model.generate_content(prompt)
            # Dummy encoding: convert response to numeric hash
            vec = np.random.rand(768).tolist()
            embeddings.append(vec)
            st.info(f"✅ Fake embedding generated for chunk {i}")
        except Exception as e:
            st.warning(f"Embedding failed: {e}")

    return embeddings

