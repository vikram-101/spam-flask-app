from flask import Flask, render_template, request
import os
import joblib

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
MODEL_PATH = os.path.join(BASE_DIR, "NB_spam_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

clf = joblib.load(MODEL_PATH)
cv = joblib.load(VEC_PATH)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message")
    data = [message]
    vect = cv.transform(data)
    prediction = clf.predict(vect)[0]
    return render_template("result.html", prediction=prediction)

# ❌ Localhost run hata diya (Render ke liye)
# Render gunicorn se app run karega

