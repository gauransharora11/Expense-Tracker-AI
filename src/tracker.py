# src/tracker.py

from src.database import get_connection
from src.ml.predict import predict_category


def add_expense(description, amount):
    category = predict_category(description)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO expenses (description, amount, category) VALUES (?, ?, ?)",
        (description, amount, category)
    )

    conn.commit()
    conn.close()

    print(f"Expense added under category: {category}")
