# 📄 Resume Classifier with Flask & Docker

This project is a **Resume Category Classification API** that allows users to classify resumes into job categories using a trained **LinearSVC model** with **TF-IDF features**. The API is built with **Flask** and it can be containerized with **Docker** for easy deployment.
---

## **Features**
- Predicts job category from resume text.
- Cleans and preprocesses resume text before prediction.
- Returns **predicted category**, **confidence score**, and cleaned text.
- Simple REST API with:
  - `GET /` → Homepage message
  - `POST /predict` → Predict category from resume text
  - Docker-ready for easy deployment anywhere.
---
## **Folder Structure**
resume_app/
│── app.py # Flask API
│── resume_analy.pkl # Trained LinearSVC model
│── tfidf_vectorizer.pkl # TF-IDF vectorizer
│── requirements.txt # Python dependencies
│── Dockerfile # Docker image
│── .dockerignore # Files to ignore in Docker build

## **Installing Dependencies**

pip install -r requirements.txt
\
## **Preprocessing**
Removes emails, phone numbers, and URLs.
Converts text to lowercase and removes special characters.
Spell checks words and removes consecutive duplicates.
Same preprocessing used during model training.
\
## **Notes**
The Flask app is for development only. For production, use a WSGI server (e.g., Gunicorn).
Docker image includes only the API and trained model, not raw CSVs or notebooks.
Confidence score is available if the model supports decision_function.


## **License**
MIT License
