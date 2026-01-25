import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

from src.database import get_connection

def train_model():
    # 1. Read data from database
    conn = get_connection()
    df = pd.read_sql(
        "SELECT description, category FROM expenses",
        conn
    )
    conn.close()

    # 2. Create ML pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    # 3. Train model
    model.fit(df["description"], df["category"])

    # 4. Save model
    joblib.dump(model, "models/expense_classifier.pkl")

    print("ML model trained and saved ✅")

if __name__ == "__main__":
    train_model()
