# src/ml/train.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

from src.ml.training_data import get_training_data

def train():
    texts, labels = get_training_data()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X, labels)

    os.makedirs("model", exist_ok=True)

    # 🔴 SAVE BOTH TOGETHER (CRITICAL)
    joblib.dump((model, vectorizer), "model/model.pkl")

    print("✅ ML model trained and saved")

if __name__ == "__main__":
    train()
