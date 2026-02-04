import os
import joblib
import numpy as np


# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")

pipeline = joblib.load(MODEL_PATH)


# -----------------------------
# CONFIDENCE LABEL
# -----------------------------
def confidence_label(score):
    if score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "medium"
    else:
        return "low"


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_expense(text, threshold=0.45):
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    best_idx = np.argmax(probs)
    best_class = classes[best_idx]
    best_score = probs[best_idx]

    # fallback to "other"
    if best_score < threshold:
        return {
            "category": "other",
            "confidence": confidence_label(best_score),
            "score": round(float(best_score), 3),
            "reason": "Low confidence → fallback to other"
        }

    return {
        "category": best_class,
        "confidence": confidence_label(best_score),
        "score": round(float(best_score), 3),
        "reason": "Predicted using TF-IDF (word + char)"
    }


# -----------------------------
# INTERACTIVE INPUT
# -----------------------------
if __name__ == "__main__":
    print("\n💰 Expense Category Predictor")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Enter expense description: ").strip()

        if user_input.lower() == "exit":
            print("👋 Exiting...")
            break

        if user_input == "":
            print("⚠️ Please enter some text\n")
            continue

        result = predict_expense(user_input)

        print("\n📊 Prediction Result")
        print(f"Category   : {result['category']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Score      : {result['score']}")
        print(f"Reason     : {result['reason']}")
        print("-" * 40)
