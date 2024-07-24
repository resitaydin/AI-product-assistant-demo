
from flask import Flask, request, jsonify
import os
from openai import OpenAI
from flask_cors import CORS  # Import the CORS class

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

# Initialize OpenAI client with API key from environment variable
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

@app.route('/ask', methods=['POST'])
def ask_openai():
    data = request.json
    user_message = data.get('message')
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": data.get('referer', ''),
                "X-Title": data.get('title', ''),
            },
            model="meta-llama/llama-3-8b-instruct:free",
            messages=[
                {
                    "role": "system",
                    "content": "You will act like a fake X Cosmetics moisturizing cream AI, focused on enhancing the consumer experience by providing detailed information about X Cosmetics moisturizing cream, its formula, testing processes and more. Do not keep the answers too long."
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )
        response_message = completion.choices[0].message.content
        return jsonify({"response": response_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
