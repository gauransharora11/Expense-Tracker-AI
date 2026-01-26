# src/ml/train.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.ml.training_data import texts, labels


def train():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # SAVE BOTH model AND vectorizer
    joblib.dump((model, vectorizer), "model/model.pkl")

    print("ML model trained and saved ✅")


if __name__ == "__main__":
    train()
