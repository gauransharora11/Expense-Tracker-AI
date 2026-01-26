# ExpenseTrack.py

from src.tracker import add_expense

desc = input("Enter expense description: ")
amt = float(input("Enter amount: "))

add_expense(desc, amt)
