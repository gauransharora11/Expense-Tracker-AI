import joblib
from pathlib import Path

MODEL_PATH = "model/model.pkl"

def load_model():
    if not Path(MODEL_PATH).exists():
        from src.ml.train import train_model
        train_model()

    return joblib.load(MODEL_PATH)

model, vectorizer = load_model()

def predict_category(text):
    X = vectorizer.transform([text.lower()])
    probs = model.predict_proba(X)[0]

    if probs.max() < 0.30:
        return "other"

    return model.classes_[probs.argmax()]
