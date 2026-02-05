import os, joblib, sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import speech_recognition as sr

st.set_page_config(page_title="Smart Expense AI", page_icon="💸", layout="wide")

# ---------------- THEME ----------------
st.markdown("""
<style>
body {background-color:#0F172A!important;}
.card {backdrop-filter: blur(14px);background: rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.1);border-radius:18px;padding:20px;
box-shadow:0 8px 32px rgba(0,0,0,0.4);margin-bottom:15px;}
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

# ---------------- DATABASE ----------------
conn = sqlite3.connect("expenses.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS expenses (
id INTEGER PRIMARY KEY AUTOINCREMENT,
date TEXT, description TEXT, category TEXT, amount REAL, confidence TEXT)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS corrections (description TEXT, correct_category TEXT)""")
conn.commit()

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier.pkl")
pipeline = joblib.load(MODEL_PATH)

# ---------------- AI HELPERS ----------------
def confidence_label(score):
    return "High" if score>=0.75 else "Medium" if score>=0.5 else "Low"

def predict_expense(text):
    probs=pipeline.predict_proba([text])[0]
    idx=np.argmax(probs)
    return pipeline.classes_[idx], confidence_label(probs[idx]), probs[idx]

def voice_to_text():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio=r.listen(source)
        try: return r.recognize_google(audio)
        except: return ""

CATEGORY_EMOJI={"food":"🍔","travel":"🚕","shopping":"🛍️","entertainment":"🎬","other":"📦"}

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Menu")
page=st.sidebar.radio("Navigate",["➕ Add Expense","📊 Dashboard","📜 History"])

# ---------------- ADD EXPENSE ----------------
if page=="➕ Add Expense":
    st.markdown('<div class="big-title">Add Expense</div>',unsafe_allow_html=True)

    if st.button("🎙 Use Voice"):
        st.session_state.voice_text=voice_to_text()

    desc=st.text_input("Description",value=st.session_state.get("voice_text",""))
    amount=st.number_input("Amount",0.0)
    date=st.date_input("Date",datetime.today())

    if st.button("Predict & Save"):
        cat,conf,score=predict_expense(desc)
        cursor.execute("INSERT INTO expenses VALUES(NULL,?,?,?,?,?)",(str(date),desc,cat,amount,conf))
        conn.commit()
        st.success(f"Saved as {cat}")
        st.progress(min(score,1.0))

# ---------------- DASHBOARD ----------------
elif page=="📊 Dashboard":
    df=pd.read_sql_query("SELECT * FROM expenses",conn)
    if df.empty: st.info("No data yet.")
    else:
        df["date"]=pd.to_datetime(df["date"])

        # FILTERS
        st.sidebar.subheader("Filters")
        start,end=st.sidebar.date_input("Date Range",[df["date"].min(),df["date"].max()])
        min_amt,max_amt=st.sidebar.slider("Amount Range",0.0,float(df["amount"].max()),(0.0,float(df["amount"].max())))

        df=df[(df["date"]>=pd.to_datetime(start))&(df["date"]<=pd.to_datetime(end))]
        df=df[(df["amount"]>=min_amt)&(df["amount"]<=max_amt)]

        col1,col2,col3=st.columns(3)
        col1.metric("Total Spent",f"₹{df['amount'].sum():.2f}")
        col2.metric("Transactions",len(df))
        col3.metric("Top Category",df['category'].value_counts().idxmax())

        st.plotly_chart(px.bar(df.groupby("category")["amount"].sum().reset_index(),
                               x="category",y="amount",color="category"),use_container_width=True)

        # PIE CHART
        st.plotly_chart(px.pie(df,names="category",values="amount",title="Spending Distribution"),
                        use_container_width=True)

# ---------------- HISTORY ----------------
elif page=="📜 History":
    st.markdown('<div class="big-title">Expense History</div>',unsafe_allow_html=True)

    df=pd.read_sql_query("SELECT * FROM expenses",conn)
    search=st.text_input("🔍 Search description")

    if search:
        df=df[df["description"].str.contains(search,case=False)]

    for _,row in df.iterrows():
        with st.container():
            c1,c2,c3,c4,c5,c6,c7=st.columns([1,2,1,1,1,1,1])
            c1.write(row["date"])
            c2.write(row["description"])
            c3.write(row["category"])
            c4.write(f"₹{row['amount']:.2f}")
            c5.write(row["confidence"])

            if c6.button("✏ Edit",key=f"edit{row['id']}"):
                st.session_state.edit_id=row["id"]

            if c7.button("🗑",key=f"del{row['id']}"):
                cursor.execute("DELETE FROM expenses WHERE id=?",(row["id"],))
                conn.commit(); st.rerun()

    # EDIT FORM
    if "edit_id" in st.session_state:
        edit_df=pd.read_sql_query(f"SELECT * FROM expenses WHERE id={st.session_state.edit_id}",conn)
        if not edit_df.empty:
            e=edit_df.iloc[0]
            st.markdown("### ✏ Edit Entry")
            new_desc=st.text_input("Description",e["description"])
            new_amt=st.number_input("Amount",value=float(e["amount"]))
            new_cat=st.selectbox("Category",list(CATEGORY_EMOJI.keys()),index=list(CATEGORY_EMOJI.keys()).index(e["category"]))
            if st.button("Update"):
                cursor.execute("UPDATE expenses SET description=?,amount=?,category=? WHERE id=?",
                               (new_desc,new_amt,new_cat,e["id"]))
                conn.commit(); del st.session_state.edit_id; st.rerun()

    st.download_button("Download CSV",df.to_csv(index=False).encode(),"expenses.csv")
