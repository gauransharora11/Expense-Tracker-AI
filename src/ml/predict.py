import joblib
import numpy as np

# Load trained model
model, vectorizer = joblib.load("model/model.pkl")

def predict_category(text):
    X = vectorizer.transform([text.lower()])
    probs = model.predict_proba(X)[0]

    max_prob = probs.max()
    predicted_class = model.classes_[np.argmax(probs)]

    if max_prob < 0.0:
        return "other"

    return predicted_class


# 🔥 THIS PART WAS MISSING (OUTPUT)
if __name__ == "__main__":
    user_text = input("Enter expense description: ")
    result = predict_category(user_text)
    print("Predicted Category:", result)
