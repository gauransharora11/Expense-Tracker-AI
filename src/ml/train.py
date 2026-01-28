import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DATA_FILES = [
    "data/food.csv",
    "data/travel.csv",
    "data/shopping.csv",
    "data/entertainment.csv"
]

MODEL_PATH = "model/model.pkl"

def train_model():
    dfs = []

    for file in DATA_FILES:
        if Path(file).exists():
            dfs.append(pd.read_csv(file))

    if not dfs:
        raise Exception("❌ No CSV files found")

    data = pd.concat(dfs, ignore_index=True)

    X = data["text"]
    y = data["category"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    Path("model").mkdir(exist_ok=True)
    joblib.dump((model, vectorizer), MODEL_PATH)

    print("✅ Model auto-trained and saved")

if __name__ == "__main__":
    train_model()
