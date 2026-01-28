import joblib
import numpy as np

model, vectorizer = joblib.load("model/model.pkl")

def predict_category(text):
    X = vectorizer.transform([text.lower()])
    probs = model.predict_proba(X)[0]

    if probs.max() < 0.0:
        return "other"

    return model.classes_[probs.argmax()]


if __name__ == "__main__":
    text = input("Enter expense description: ")
    print("Predicted Category:", predict_category(text))
