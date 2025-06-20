from babyagi.rag.rag_store import embed_text
print(embed_text(["Hello world"]))  # ✅ should return a 1536-length vector
