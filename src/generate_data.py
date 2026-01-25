import random
from src.database import get_connection

# Sample expense data
sample_expenses = [
    ("Dominos Pizza", 250, "Food"),
    ("Burger King", 180, "Food"),
    ("Uber Ride", 320, "Transport"),
    ("Ola Cab", 280, "Transport"),
    ("Electricity Bill", 1200, "Utilities"),
    ("Mobile Recharge", 399, "Utilities"),
    ("Movie Ticket", 350, "Entertainment"),
    ("Netflix Subscription", 499, "Entertainment"),
    ("Amazon Shopping", 1500, "Shopping"),
    ("Flipkart Order", 2200, "Shopping")
]

def generate_expenses(n=200):
    conn = get_connection()
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY,
            description TEXT,
            amount REAL,
            category TEXT
        )
    """)

    for _ in range(n):
        desc, amount, category = random.choice(sample_expenses)
        cur.execute(
            "INSERT INTO expenses (description, amount, category) VALUES (?, ?, ?)",
            (desc, amount, category)
        )

    conn.commit()
    conn.close()
    print(f"{n} sample expenses added ✅")

if __name__ == "__main__":
    generate_expenses()
