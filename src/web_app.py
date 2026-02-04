import os
import joblib
import numpy as np
import streamlit as st
import speech_recognition as sr

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at {MODEL_PATH}. Please run train.py first.")
    st.stop()

pipeline = joblib.load(MODEL_PATH)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def confidence_label(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.50:
        return "Medium"
    else:
        return "Low"

def predict_expense(text, threshold=0.45):
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    best_idx = np.argmax(probs)
    best_class = classes[best_idx]
    best_score = probs[best_idx]

    if best_score < threshold:
        return "other", confidence_label(best_score), best_score, "Low confidence → fallback to other"

    return best_class, confidence_label(best_score), best_score, "Predicted using TF-IDF (word + char)"


def get_voice_input():
    """Listens to the microphone and returns text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...") # Prints to terminal for debugging
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.warning("❌ Could not understand audio.")
        except sr.RequestError:
            st.error("❌ API unavailable.")
        except sr.WaitTimeoutError:
            st.warning("❌ No speech detected.")
        except Exception as e:
            st.error(f"Error: {e}")
    return None


# --- NEW: CALLBACK FUNCTION ---
def handle_voice_update():
    """Run this BEFORE the page reload to safely update state."""
    voice_text = get_voice_input()
    if voice_text:
        st.session_state['expense_text'] = voice_text


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Expense Category AI", page_icon="💰")
st.title("💰 Expense Category Predictor")

# Initialize session state if it doesn't exist
if 'expense_text' not in st.session_state:
    st.session_state['expense_text'] = ""

# Layout
col1, col2 = st.columns([4, 1])

with col2:
    st.write("") 
    st.write("")
    # FIX: Use 'on_click' to trigger the update safely
    st.button("🎤 Voice", on_click=handle_voice_update)

with col1:
    # The text input will now auto-fill because the callback updated the state 
    # BEFORE this widget is rendered.
    user_input = st.text_input(
        "Expense description", 
        key="expense_text", 
        placeholder="eg: adidas shoes from mall"
    )

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an expense description.")
    else:
        category, confidence, score, reason = predict_expense(user_input)

        st.divider()
        st.subheader(f"🏷️ Category: {category}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", confidence)
        c2.metric("Score", f"{round(score, 3)}")
        c3.caption(reason)