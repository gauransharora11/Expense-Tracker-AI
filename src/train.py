import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "expenses.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# LOAD DATA & FIX COLUMNS (UPDATED)
# -----------------------------
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: File not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

# 1. Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# 2. Check for 'text' column
# Use 'text' if present, otherwise look for alternatives
if "text" not in df.columns:
    if "Description" in df.columns and "category" in df.columns:
        # If we have category, then Description is likely the text
        print("⚠️ Column 'text' missing. Mapping 'Description' -> 'text'")
        df.rename(columns={"Description": "text"}, inplace=True)
    else:
        raise KeyError(f"❌ Could not find a 'text' column. Available: {df.columns.tolist()}")

# 3. Ensure 'category' column exists (target variable)
if "category" not in df.columns:
    # >>> THIS IS THE FIX <<<
    # Your CSV uses 'Description' for the category labels
    if "Description" in df.columns:
         print("⚠️ Column 'category' missing. Mapping 'Description' -> 'category'")
         df.rename(columns={"Description": "category"}, inplace=True)
    elif "Category" in df.columns:
         df.rename(columns={"Category": "category"}, inplace=True)
    else:
        raise KeyError(f"❌ Missing target column 'category'. Available: {df.columns.tolist()}")

# 4. Drop rows with missing values
df.dropna(subset=["text", "category"], inplace=True)

X = df["text"].astype(str)
y = df["category"].astype(str)

print(f"✅ Data loaded: {len(df)} rows")
print(f"   Example text: {X.iloc[0]}")
print(f"   Example category: {y.iloc[0]}")


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# FEATURE PIPELINE
# -----------------------------
word_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2
)

features = FeatureUnion([
    ("word_tfidf", word_tfidf),
    ("char_tfidf", char_tfidf)
])


# -----------------------------
# CLASSIFIER
# -----------------------------
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# -----------------------------
# FULL PIPELINE
# -----------------------------
pipeline = Pipeline([
    ("features", features),
    ("classifier", classifier)
])


# -----------------------------
# TRAIN
# -----------------------------
print("⏳ Training model...")
pipeline.fit(X_train, y_train)


# -----------------------------
# EVALUATE
# -----------------------------
y_pred = pipeline.predict(X_test)
print("\n" + classification_report(y_test, y_pred))


# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(pipeline, os.path.join(MODEL_DIR, "classifier.pkl"))

print(f"✅ Model saved to {os.path.join(MODEL_DIR, 'classifier.pkl')}")