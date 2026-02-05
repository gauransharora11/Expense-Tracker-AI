# ExpenseTrack.py

from src.tracker import add_expense

desc = input("Enter expense description: ")
amt = float(input("Enter amount: "))

add_expense(desc, amt)
# source venv/bin/activate
# python -m src.train
# python -m src.predict
#quit
# burger
#streamlit run src/web_app.py



