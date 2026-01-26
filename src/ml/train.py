import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train():
    # 1️⃣ Training data (example)
    texts = [
        "Bought groceries",
        "Paid electricity bill",
        "Movie ticket",
        "Dinner at restaurant",
        "Uber ride"
    ]

    labels = [
        "Food",
        "Bills",
        "Entertainment",
        "Food",
        "Travel"
    ]

    # 2️⃣ Convert text → numbers
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # 3️⃣ CREATE & TRAIN MODEL  ✅ (THIS WAS MISSING)
    model = LogisticRegression()
    model.fit(X, labels)

    # 4️⃣ Save model + vectorizer
    joblib.dump(model, "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

    print("ML model trained and saved ✅")

if __name__ == "__main__":
    train()
