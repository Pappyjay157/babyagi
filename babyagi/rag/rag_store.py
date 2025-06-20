import os
import faiss
import tiktoken
import numpy as np
from typing import List, Tuple
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)
stored_chunks: List[str] = []

tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    words = text.split()
    chunks, current_chunk, token_count = [], [], 0

    for word in words:
        token_count += len(tokenizer.encode(word))
        current_chunk.append(word)
        if token_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, token_count = [], 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def embed_text(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def store_document_chunks(text: str):
    chunks = chunk_text(text)
    embeddings = embed_text(chunks)

    global stored_chunks
    stored_chunks.extend(chunks)

    embeddings_array = np.array(embeddings, dtype="float32")
    if embeddings_array.shape[0] > 0:
        index.add(embeddings_array)

def query_similar_chunks(question: str, k: int = 5) -> List[Tuple[str, float]]:
    question_embedding = embed_text([question])[0]
    D, I = index.search(np.array([question_embedding], dtype="float32"), k)
    return [(stored_chunks[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(stored_chunks)]
