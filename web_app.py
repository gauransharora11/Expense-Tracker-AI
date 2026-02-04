import streamlit as st
from src.train import train_model
from src.ml.predict import predict_category

st.set_page_config(page_title="Expense Tracker AI", layout="wide")

st.title("💼 Expense Tracker AI")

# Sidebar
with st.sidebar:
    st.header("⚙ Controls")

    if st.button("🔄 Retrain Model"):
        with st.spinner("Training model..."):
            train_model()
        st.success("Model retrained successfully!")

# Input
st.subheader("Enter Expense Description")
text = st.text_input("Example: KFC bucket, Uber ride, Amazon shoes")

if text:
    category = predict_category(text)
    st.success(f"Predicted Category: **{category.upper()}**")
