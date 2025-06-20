# babyagi/api/route.py

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from babyagi.rag.rag_store import store_document_chunks, query_similar_chunks
from openai import OpenAI
import os

api = Blueprint("student_api", __name__, url_prefix="/api")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 📤 Route to upload and embed notes
@api.route("/upload", methods=["POST"])
def upload_notes():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)

    if not filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files are supported"}), 400

    try:
        content = file.read().decode("utf-8")
        store_document_chunks(content)
        return jsonify({"message": f"Uploaded and stored: {filename}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ❓ Route to ask a question based on notes (RAG-powered)
@api.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question = data["question"]
    try:
        # 1. Retrieve relevant chunks
        chunks = query_similar_chunks(question)
        context_text = "\n\n".join([chunk for chunk, _ in chunks])

        # 2. Use GPT-4 to generate the answer
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for students. Use the notes to answer clearly."
                },
                {
                    "role": "user",
                    "content": f"Here are the notes:\n{context_text}\n\nQuestion: {question}"
                }
            ]
        )
        answer = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
