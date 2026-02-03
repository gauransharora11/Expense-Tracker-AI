# src/ml/training_data.py

def get_training_data():
    data = {
        "food": [
            "kfc food", "dominos pizza", "mcdonalds burger",
            "swiggy order", "zomato food", "restaurant bill",
            "biryani dinner", "coffee cafe", "starbucks coffee"
        ],
        "travel": [
            "uber ride", "patola cab", "metro ticket",
            "flight booking", "train ticket", "bus fare"
        ],
        "shopping": [
            "amazon shopping", "flipkart order",
            "zara clothes", "myntra shoes",
            "mobile purchase", "electronics store"
        ],
        "entertainment": [
            "movie ticket", "pvr cinema",
            "netflix subscription", "spotify music",
            "concert ticket", "game purchase"
        ],
        "other": [
            "random payment", "misc expense",
            "unknown transaction", "cash transfer"
        ]
    }

    texts, labels = [], []
    for label, items in data.items():
        for text in items:
            texts.append(text)
            labels.append(label)

    return texts, labels
