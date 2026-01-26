print("predict.py started")

import joblib

def predict_category(text):
    model, vectorizer = joblib.load("model/model.pkl")
    X = vectorizer.transform([text])
    return model.predict(X)[0]

if __name__ == "__main__":
    print("inside main")
    user_text = input("Enter expense description: ")
    print("Predicted Category:", predict_category(user_text))
