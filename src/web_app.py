import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Expense AI", page_icon="💸", layout="wide")

# -----------------------------
# MODERN UI STYLING
# -----------------------------
st.markdown("""
<style>
body {background-color:#0f172a;}
.big-title {font-size:34px; font-weight:700;}
.card {
    padding:20px;
    border-radius:14px;
    background: linear-gradient(145deg,#1e293b,#0f172a);
    box-shadow:0 4px 12px rgba(0,0,0,0.3);
    margin-bottom:15px;
}
button[kind="primary"] {
    background-color:#6366f1 !important;
    border-radius:8px !important;
    height:3em;
    transition:0.3s;
}
button[kind="primary"]:hover {transform:scale(1.05);}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATABASE SETUP (PERMANENT STORAGE)
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
conn.commit()

# -----------------------------
# LOAD ML MODEL
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")
pipeline = joblib.load(MODEL_PATH)

# -----------------------------
# HELPERS
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
        return "other", "Low", best_score, probs
    return best_class, confidence_label(best_score), best_score, probs

CATEGORY_EMOJI = {"food":"🍔","travel":"🚕","shopping":"🛍️","entertainment":"🎬","other":"📦"}

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("💸 Smart Expense AI")
page = st.sidebar.radio("Navigate", ["➕ Add Expense", "📊 Dashboard", "📜 History"])

# -----------------------------
# ADD EXPENSE PAGE
# -----------------------------
if page == "➕ Add Expense":
    st.markdown('<div class="big-title">Add Expense with AI</div>', unsafe_allow_html=True)

    desc = st.text_input("📝 What did you spend on?")
    amount = st.number_input("💰 Amount", min_value=0.0, step=1.0)
    date = st.date_input("📅 Date", datetime.today())

    if st.button(" Predict & Save", use_container_width=True):
        if desc.strip() == "":
            st.warning("Enter description")
        else:
            category, confidence, score, probs = predict_expense(desc)
            emoji = CATEGORY_EMOJI.get(category,"❓")

            cursor.execute("INSERT INTO expenses (date, description, category, amount, confidence) VALUES (?,?,?,?,?)",
                           (str(date), desc, category, amount, confidence))
            conn.commit()

            st.success("Expense saved!")
            st.markdown(f"### {emoji} {category.upper()}")
            st.progress(min(score,1.0))

# -----------------------------
# DASHBOARD
# -----------------------------
elif page == "📊 Dashboard":
    st.markdown('<div class="big-title">Expense Dashboard</div>', unsafe_allow_html=True)

    df = pd.read_sql_query("SELECT * FROM expenses", conn)

    if df.empty:
        st.info("No data yet.")
    else:
        col1,col2,col3 = st.columns(3)
        col1.metric("Total Spent", f"${df['amount'].sum():.2f}")
        col2.metric("Transactions", len(df))
        col3.metric("Top Category", df['category'].value_counts().idxmax())

        st.markdown("### 📊 Spending by Category")
        st.bar_chart(df.groupby("category")["amount"].sum())

        st.markdown("### 🥧 Category Distribution")
        st.pyplot(df.groupby("category")["amount"].sum().plot.pie(autopct="%1.1f%%").figure)

        st.markdown("### 📈 Spending Over Time")
        df["date"] = pd.to_datetime(df["date"])
        st.line_chart(df.groupby("date")["amount"].sum())

# -----------------------------
# HISTORY
# -----------------------------
elif page == "📜 History":
    st.markdown('<div class="big-title">Expense History</div>', unsafe_allow_html=True)

    df = pd.read_sql_query("SELECT * FROM expenses", conn)

    if df.empty:
        st.info("No expenses recorded.")
    else:
        st.dataframe(df, use_container_width=True)

        # DOWNLOAD BUTTON
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download as Excel", csv, "expenses.csv", "text/csv")
