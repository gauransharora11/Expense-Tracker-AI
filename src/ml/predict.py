import joblib


def predict_category(text):
    model, vectorizer = joblib.load("model/model.pkl")

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]

    confidence = max(probs)
    predicted = model.classes_[probs.argmax()]

    if confidence < 0.55:
        return "other"

    return predicted


if __name__ == "__main__":
    user_text = input("Enter expense description: ")
    print("Predicted Category:", predict_category(user_text))
