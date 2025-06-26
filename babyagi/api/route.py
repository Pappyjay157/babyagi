# babyagi/api/route.py
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from babyagi.rag.rag_store import store_document_chunks, query_similar_chunks
import os
from openai import OpenAI

api = Blueprint("student_api", __name__, url_prefix="/api")

# Add Open api key
client = OpenAI()

# 📤 Route to upload and embed notes
@api.route("/upload", methods=["POST"])
def upload_notes():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename or "")
    if not filename:
        return jsonify({"error": "Invalid file name"}), 400

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
        content = completion.choices[0].message.content
        answer = content.strip() if content is not None else "No answer generated."
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route("/task", methods=["POST"])
def run_task():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON in request body"}), 400
    task_type = data.get("task_type")
    input_text = data.get("input_text")

    if not task_type or not input_text:
        return jsonify({"error": "Missing task_type or input_text"}), 400

    try:
        # Build dynamic prompt based on task type
        prompt = {
            "summarize": f"Summarize the following text:\n\n{input_text}",
            "draft": f"Draft a student-friendly explanation for:\n\n{input_text}",
            "answer": f"Answer the following question:\n\n{input_text}",
        }.get(task_type.lower(), f"Please process the following task:\n\n{input_text}")

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for students."},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content
        result = content.strip() if content is not None else "No result generated."
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500