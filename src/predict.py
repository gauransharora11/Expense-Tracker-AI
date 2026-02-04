import os
import joblib
import numpy as np
import sys
import speech_recognition as sr  # <--- NEW IMPORT

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("   Please run 'python -m src.train' first.")
    sys.exit(1)

pipeline = joblib.load(MODEL_PATH)


# -----------------------------
# CONFIDENCE LABELS
# -----------------------------
def confidence_label(score):
    if score >= 0.85:
        return "high"
    elif score >= 0.60:
        return "medium"
    else:
        return "low"


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_expense(text, threshold=0.40):
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    best_idx = np.argmax(probs)
    best_class = classes[best_idx]
    best_score = probs[best_idx]

    if best_score < threshold:
        return {
            "input": text,
            "category": "other",
            "confidence": confidence_label(best_score),
            "score": round(float(best_score), 3),
            "reason": "Low confidence — fallback applied"
        }

    return {
        "input": text,
        "category": best_class,
        "confidence": confidence_label(best_score),
        "score": round(float(best_score), 3),
        "reason": "Predicted using TF-IDF features"
    }


# -----------------------------
# VOICE INPUT FUNCTION (NEW)
# -----------------------------
def get_voice_input():
    recognizer = sr.Recognizer()
    
    # Try to use the default system microphone
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening... (Speak clearly now)")
            
            # Adjust for ambient noise (helps in noisy rooms)
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen (stops when you stop speaking)
            audio = recognizer.listen(source, timeout=5)
            
            print("⏳ Processing audio...")
            
            # Convert speech to text using Google's free API
            text = recognizer.recognize_google(audio)
            print(f"🗣️  You said: '{text}'")
            return text

    except sr.WaitTimeoutError:
        print("❌ No speech detected (timeout).")
        return None
    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
        return None
    except sr.RequestError:
        print("❌ Could not request results (check internet connection).")
        return None
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return None


# -----------------------------
# INTERACTIVE MODE
# -----------------------------
if __name__ == "__main__":
    print("\n🔮 Expense Classifier Loaded!")
    print("Type a description OR type 'v' to use voice.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input(">> Enter expense: ").strip()

            # Exit condition
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye! 👋")
                break
            
            # --- VOICE MODE TRIGGER ---
            if user_input.lower() in ["v", "voice", "mic"]:
                voice_text = get_voice_input()
                if voice_text:
                    user_input = voice_text
                else:
                    continue  # Skip prediction if voice failed
            # --------------------------

            if not user_input:
                continue

            # Run prediction
            prediction = predict_expense(user_input)
            print(prediction)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break