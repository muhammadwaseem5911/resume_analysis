from flask import Flask, request, jsonify
import pickle
import re
from spellchecker import SpellChecker
import numpy as np
import os

# ==========================
# 1. Load model + vectorizer
# ==========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resume_analy.pkl")
VEC_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VEC_PATH, "rb"))

# ==========================
# 2. Preprocessing
# ==========================
spell = SpellChecker()
spell.word_frequency.load_words([
    "sql","python","java","excel","ml","ai","r","c++","c#","bsc","msc","django",
    "flask","html","css","js","javascript","pandas","numpy","powerbi","hr","cv",
    "api","data","etl","machine","learning"
])

email_re = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
phone_re = re.compile(r'\+?\d[\d\-\s\(\)]{6,}\d')
url_re = re.compile(r'https?://\S+|www\.\S+')

def clean_text(text):
    if text is None:
        return ""
    text = email_re.sub(" <EMAIL> ", text)
    text = phone_re.sub(" <PHONE> ", text)
    text = url_re.sub(" <URL> ", text)
    text = re.sub(r"[^A-Za-z0-9\s\+\#\.\&\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    words = text.split()
    corrected = [spell.correction(w) if w not in spell else w for w in words]
    final = [corrected[i] for i in range(len(corrected)) if i == 0 or corrected[i] != corrected[i-1]]
    return " ".join(final)

# ==========================
# 3. Flask app
# ==========================
app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def home():
    return "✅ Resume Classifier API is running. Use POST /predict to classify resumes."

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "resume_text" not in data:
        return jsonify({"error": "Missing 'resume_text' in request"}), 400

    raw_text = data["resume_text"]
    cleaned = clean_text(raw_text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]

    # Confidence score
    if hasattr(model, "decision_function"):
        scores = model.decision_function(transformed)
        confidence = float(np.max(scores))
    else:
        confidence = None

    return jsonify({
        "predicted_category": prediction,
        "confidence_score": confidence,
        "cleaned_text": cleaned
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)