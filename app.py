from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from model import calculate_similarity

import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/analyze', methods=['POST'])
def analyze_resume():

    resume = request.files['resume']
    job_description = request.form['job_description']

    file_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    resume.save(file_path)

    # Extract text from PDF
    reader = PdfReader(file_path)

    resume_text = ""

    for page in reader.pages:
        resume_text += page.extract_text()

    score, missing_keywords = calculate_similarity(
        resume_text,
        job_description
    )

    return jsonify({
        "ats_score": score,
        "missing_keywords": missing_keywords
    })

if __name__ == '__main__':
    app.run(debug=True)