import os
import joblib
import numpy as np

# Load paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "model/model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model/vectorizer.pkl"))

LABELS = model.classes_

def predict_with_explanation(text):
    text = text.lower().strip()

    # Convert text to features
    X = vectorizer.transform([text])

    # Get probabilities
    probs = model.predict_proba(X)[0]
    best_index = np.argmax(probs)
    best_label = LABELS[best_index]
    best_prob = probs[best_index]

    # Check if input is weak
    if X.nnz < 3 or best_prob < 0.55:
        return {
            "input": text,
            "prediction": "other",
            "confidence": "low",
            "reason": "Too little information or unseen input"
        }

    # Confidence level
    confidence = (
        "high" if best_prob >= 0.75 else "medium"
    )

    # Explainability
    feature_names = vectorizer.get_feature_names_out()
    coef = model.base_estimator.coef_[best_index]

    top_features_idx = np.argsort(coef)[-5:]
    top_features = [feature_names[i] for i in top_features_idx]

    return {
        "input": text,
        "prediction": best_label,
        "confidence": confidence,
        "probability": round(best_prob, 3),
        "important_features": top_features,
        "explanation": (
            f"The model detected these signals: {top_features}, "
            f"which are associated with {best_label}"
        )
    }

# CLI loop
if __name__ == "__main__":
    while True:
        text = input("Enter expense description: ")
        if text.lower() in {"exit", "quit"}:
            break
        result = predict_with_explanation(text)
        print(result)
