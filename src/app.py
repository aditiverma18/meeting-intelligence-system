from flask import Flask, request, jsonify
from flask_cors import CORS
from analyze_meeting import analyze_meeting

app = Flask(__name__)
CORS(app)

@app.route("/api/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    transcript = data.get("transcript", "")

    if not transcript:
        return jsonify({"error": "Transcript is required"}), 400

    result = analyze_meeting(transcript)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)