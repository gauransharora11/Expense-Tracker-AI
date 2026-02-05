import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
from datetime import datetime
import plotly.express as px
import speech_recognition as sr

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Expense AI", page_icon="💸", layout="wide")

# -----------------------------
# ANIMATED PRO THEME
# -----------------------------
st.markdown("""
<style>

/* DARK BACKGROUND ONLY CHANGE */
body {
    background-color: #0F172A !important;
}

/* keep everything else exactly same below */
@keyframes gradientBG {0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
.card {backdrop-filter: blur(14px); background: rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.1); border-radius:18px; padding:20px;
box-shadow:0 8px 32px rgba(0,0,0,0.4); margin-bottom:15px;}
.big-title {font-size:34px;font-weight:700;margin-bottom:10px;}
button[kind="primary"] {background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;border-radius:12px!important;height:3em;}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="padding:25px;border-radius:18px;background:linear-gradient(90deg,#6366f1,#8b5cf6);
color:white;text-align:center;font-size:26px;font-weight:600;margin-bottom:25px;">
💸 Smart Expense AI — Voice + Learning Edition
</div>
""", unsafe_allow_html=True)

# -----------------------------
# DATABASE
# -----------------------------
conn = sqlite3.connect("expenses.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS expenses (
id INTEGER PRIMARY KEY AUTOINCREMENT,
date TEXT,
description TEXT,
category TEXT,
amount REAL,
confidence TEXT
)
""")

# TABLE FOR AI CORRECTIONS
cursor.execute("""
CREATE TABLE IF NOT EXISTS corrections (
description TEXT,
correct_category TEXT
)
""")
conn.commit()

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")
pipeline = joblib.load(MODEL_PATH)

# -----------------------------
# AI HELPERS
# -----------------------------
def confidence_label(score):
    if score >= 0.75: return "High"
    elif score >= 0.50: return "Medium"
    else: return "Low"

def predict_expense(text, threshold=0.45):
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    idx = np.argmax(probs)
    best_class, best_score = classes[idx], probs[idx]
    if best_score < threshold:
        return "other", "Low", best_score
    return best_class, confidence_label(best_score), best_score

def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            return ""

CATEGORY_EMOJI = {"food":"🍔","travel":"🚕","shopping":"🛍️","entertainment":"🎬","other":"📦"}

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Navigate", ["➕ Add Expense", "📊 Dashboard", "📜 History"])

# -----------------------------
# ADD EXPENSE PAGE
# -----------------------------
if page == "➕ Add Expense":
    st.markdown('<div class="big-title">Add Expense</div>', unsafe_allow_html=True)

    if st.button("🎙 Use Voice"):
        st.session_state.voice_text = voice_to_text()

    desc = st.text_input("📝 Description", value=st.session_state.get("voice_text",""))
    amount = st.number_input("💰 Amount", min_value=0.0, step=1.0)
    date = st.date_input("📅 Date", datetime.today())

    if st.button("🔮 Predict & Save"):
        if desc.strip()=="":
            st.warning("Enter description")
        else:
            category, confidence, score = predict_expense(desc)
            emoji = CATEGORY_EMOJI.get(category,"❓")

            cursor.execute("INSERT INTO expenses VALUES (NULL,?,?,?,?,?)",
                           (str(date),desc,category,amount,confidence))
            conn.commit()

            st.success(f"{emoji} Saved as {category}")
            st.progress(min(score,1.0))

            # ---------------- AUTO LEARNING CORRECTION
            st.markdown("### ❓ Wrong Category?")
            correct = st.selectbox("Select correct category", list(CATEGORY_EMOJI.keys()))
            if st.button("Submit Correction"):
                cursor.execute("INSERT INTO corrections VALUES (?,?)",(desc,correct))
                conn.commit()
                st.success("Correction saved! AI will improve next training.")

# -----------------------------
# DASHBOARD
# -----------------------------
elif page == "📊 Dashboard":
    df = pd.read_sql_query("SELECT * FROM expenses", conn)
    if df.empty:
        st.info("No data yet.")
    else:
        df["date"]=pd.to_datetime(df["date"])
        col1,col2,col3=st.columns(3)
        col1.metric("Total Spent",f"${df['amount'].sum():.2f}")
        col2.metric("Transactions",len(df))
        col3.metric("Top Category",df['category'].value_counts().idxmax())

        st.plotly_chart(px.bar(df.groupby("category")["amount"].sum().reset_index(),
                               x="category",y="amount",color="category"),use_container_width=True)

# -----------------------------
# HISTORY
# -----------------------------
elif page == "📜 History":
    df=pd.read_sql_query("SELECT * FROM expenses",conn)
    if df.empty:
        st.info("No records.")
    else:
        st.dataframe(df,use_container_width=True)
        csv=df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV",csv,"expenses.csv","text/csv")
