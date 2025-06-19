import babyagi
import os
from flask import Flask
from dotenv import load_dotenv
from babyagi.api.route import api




load_dotenv()
app = babyagi.create_app('/dashboard')

app.register_blueprint(api)

# Add OpenAI key to enable automated descriptions and embedding of functions.
babyagi.add_key_wrapper('openai_api_key',os.environ['OPENAI_API_KEY'])


@app.route('/')
def home():
    return f"Welcome to the main app. Visit <a href=\"/dashboard\">/dashboard</a> for BabyAGI dashboard."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
