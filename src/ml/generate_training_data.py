# src/ml/generate_training_data.py

import random

food = [
    "pizza", "burger", "kfc", "dominos", "mcdonalds",
    "swiggy", "zomato", "biryani", "coffee", "starbucks"
]

travel = [
    "uber", "ola", "metro", "bus", "flight",
    "train", "cab", "auto", "taxi"
]

shopping = [
    "amazon", "flipkart", "mall", "clothes",
    "shoes", "electronics", "shopping"
]

entertainment = [
    "netflix", "spotify", "movie", "cinema",
    "prime video", "hotstar", "concert"
]

texts = []
labels = []

def add_samples(words, label, count):
    for _ in range(count):
        texts.append(random.choice(words))
        labels.append(label)

add_samples(food, "Food", 500)
add_samples(travel, "Travel", 500)
add_samples(shopping, "Shopping", 500)
add_samples(entertainment, "Entertainment", 500)

print(f"Generated {len(texts)} samples")

with open("src/ml/training_data.py", "w") as f:
    f.write("texts = " + str(texts) + "\n\n")
    f.write("labels = " + str(labels))
