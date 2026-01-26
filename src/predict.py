import joblib

def predict():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")

    text = input("Enter expense description: ")

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    print("Predicted Category:", prediction[0])

if __name__ == "__main__":
    predict()
