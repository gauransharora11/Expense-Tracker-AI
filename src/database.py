import sqlite3

DB_PATH = "data/expenses.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def insert_expense(description, amount, category):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO expenses (description, amount, category)
        VALUES (?, ?, ?)
    """, (description, amount, category))

    conn.commit()
    conn.close()


def fetch_expenses(limit=10, offset=0):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT description, amount, category
        FROM expenses
        ORDER BY rowid DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cur.fetchall()
    conn.close()
    return rows
