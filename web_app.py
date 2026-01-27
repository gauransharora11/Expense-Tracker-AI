import streamlit as st
from src.database import insert_expense, fetch_expenses
from src.ml.predict import predict_category

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Expense Tracker AI",
    page_icon="💸",
    layout="centered"
)

# ----------------- STYLES -----------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
}
.sub-title {
    color: gray;
}
.card {
    padding: 12px;
    border-radius: 10px;
    background-color: #f8f9fa;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.markdown("<div class='main-title'>💸 Expense Tracker AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Smart expense tracking with ML-powered categorization</div>", unsafe_allow_html=True)
st.divider()

# ----------------- ADD EXPENSE -----------------
st.subheader("➕ Add New Expense")

desc = st.text_input("Expense Description")
amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0)

if st.button("Add Expense"):
    if desc.strip() == "":
        st.warning("Please enter a description")
    else:
        category = predict_category(desc)
        insert_expense(desc, amount, category)
        st.success(f"Saved as **{category}**")

st.divider()

# ----------------- RECENT EXPENSES -----------------
st.subheader("📊 Recent Expenses")

if "page" not in st.session_state:
    st.session_state.page = 0

LIMIT = 10
OFFSET = st.session_state.page * LIMIT

rows = fetch_expenses(LIMIT, OFFSET)

if rows:
    for desc, amt, cat in rows:
        st.markdown(
            f"""
            <div class="card">
                <b>{desc}</b><br>
                ₹{amt} — <i>{cat}</i>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("No more records")

# ----------------- PAGINATION -----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ Previous") and st.session_state.page > 0:
        st.session_state.page -= 1

with col2:
    if st.button("Next ➡"):
        st.session_state.page += 1
