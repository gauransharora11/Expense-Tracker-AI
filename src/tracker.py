from src.database import get_connection

def add_expense():
    # 1. Take input from user
    amount = float(input("Enter amount: "))
    category = input("Enter category: ")

    # 2. Connect to database
    conn = get_connection()
    cur = conn.cursor()

    # 3. Create table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY,
            amount REAL,
            category TEXT
        )
    """)

    # 4. Insert user data
    cur.execute(
        "INSERT INTO expenses (amount, category) VALUES (?, ?)",
        (amount, category)
    )

    conn.commit()
    conn.close()

    print("Expense added successfully ✅")
