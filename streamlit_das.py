import streamlit as st
import pickle
import re
import numpy as np
from spellchecker import SpellChecker
# 1. LOAD MODEL + VECTORIZER
MODEL_PATH = "resume_analy.pkl"
VEC_PATH = "tfidf_vectorizer.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VEC_PATH, "rb"))

# 2. SAME PREPROCESSING USED IN TRAINING
spell = SpellChecker()
spell.word_frequency.load_words([
    "sql","python","java","excel","ml","ai","r","c++","c#","bsc","msc","django",
    "flask","html","css","js","javascript","pandas","numpy","powerbi","hr","cv",
    "api","data","etl","machine","learning"
])

email_re = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
phone_re = re.compile(r'\+?\d[\d\-\s\(\)]{6,}\d')
url_re = re.compile(r'https?://\S+|www\.\S+')

def clean_resume(text):
    if text is None:
        return ""

    # 1. Replace email, phone, URL
    text = email_re.sub(" <EMAIL> ", text)
    text = phone_re.sub(" <PHONE> ", text)
    text = url_re.sub(" <URL> ", text)

    # 2. Basic cleanup
    text = re.sub(r"[^A-Za-z0-9\s\+\#\.\&\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    # 3. Spell correction
    words = text.split()
    corrected = [spell.correction(w) if w not in spell else w for w in words]

    # 4. Remove consecutive duplicates
    final = [corrected[i] for i in range(len(corrected)) if i == 0 or corrected[i] != corrected[i-1]]

    return " ".join(final)

# ==============================
# 3. STREAMLIT UI
# ==============================
st.set_page_config(page_title="Resume Category Classifier", layout="wide")

st.title("📄 Resume Category Classification")
st.write("Paste your resume text below. The model will classify it into the correct job category.")

resume_text = st.text_area(
    "Paste Resume Here",
    height=300,
    placeholder="Copy/paste your full resume text..."
)

if st.button("Predict Category"):
    if not resume_text.strip():
        st.warning("Please paste your resume first.")
    else:
        # Clean text
        cleaned = clean_resume(resume_text)

        # Vectorize
        transformed = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(transformed)[0]

        # Confidence score
        if hasattr(model, "decision_function"):
            scores = model.decision_function(transformed)
            confidence = np.max(scores)
        else:
            confidence = None

        st.subheader("🔍 Predicted Category:")
        st.success(prediction)

        if confidence is not None:
            st.info(f"Confidence Score: **{confidence:.3f}**")

        st.write("---")
        st.write("### 📝 Cleaned Resume Text (Used by Model)")
        st.code(cleaned)
