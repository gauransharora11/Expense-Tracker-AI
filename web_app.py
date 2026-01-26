import streamlit as st

from src.tracker import add_expense
from src.database import get_connection


st.set_page_config(page_title="Expense Tracker AI", layout="centered")

st.title("💰 Expense Tracker AI")

st.subheader("Add New Expense")

description = st.text_input("Expense description")
amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0)

if st.button("Add Expense"):
    if description and amount > 0:
        add_expense(description, amount)
        st.success("Expense added successfully!")
    else:
        st.error("Please enter valid data")

st.divider()

st.subheader("📊 Expense History")

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT description, amount, category FROM expenses")
rows = cur.fetchall()
conn.close()

if rows:
    st.table(rows)
else:
    st.info("No expenses found")
