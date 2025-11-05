import google.generativeai as genai
import streamlit as st
import os

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
EMBED_MODEL = "models/embedding-001"

def get_gemini_embedding(chunks):
    print(f"🔍 Total chunks received: {len(chunks)}")

    embeddings = []
    for i, text in enumerate(chunks):
        try:
            if not text.strip():
                print(f"⚠️ Skipping empty chunk #{i}")
                continue

            print(f"🟢 Sending chunk #{i}: {text[:100]}...")
            response = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
                title=f"Chunk-{i+1}"
            )
            print("🟣 Gemini response keys:", response.keys())

            emb = response.get("embedding", [])
            if emb:
                embeddings.append(emb)
                print(f"✅ Got embedding #{i} length {len(emb)}")
            else:
                print(f"⚠️ Empty embedding returned for chunk #{i}")

        except Exception as e:
            print(f"❌ Embedding failed for chunk #{i}: {e}")

    print(f"🔵 Total embeddings generated: {len(embeddings)} / {len(chunks)}")
    return embeddings

