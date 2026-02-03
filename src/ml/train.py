import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
DATASET_PATH = "expense_dataset_25000.csv"

df = pd.read_csv(DATASET_PATH)

# Normalize text
df["text"] = df["text"].astype(str).str.lower().str.strip()
df["category"] = df["category"].astype(str)

# -----------------------------
# 2. Create X and y  ✅ FIX
# -----------------------------
X = df["text"]
y = df["category"]

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Build ML pipeline
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    ))
])

# -----------------------------
# 5. Train model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate
# -----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 7. Save model
# -----------------------------
joblib.dump(model, "model/model.pkl")

print("\nML model trained and saved ✅")
