import pandas as pd

# Load dataset
df = pd.read_csv("expense_dataset_25000.csv")

print("Total rows:", len(df))
print("\nCategory counts:")
print(df["category"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# ✅ FIX: use 'text' column instead of 'description'
df["text"] = df["text"].astype(str).str.lower().str.strip()
df["category"] = df["category"].astype(str).str.lower().str.strip()

# Remove empty text
df = df[df["text"] != ""]

# Keep only valid categories
valid_categories = ["food", "travel", "shopping", "entertainment", "other"]
df = df[df["category"].isin(valid_categories)]

print("\nAfter cleaning:", len(df))

# Save cleaned dataset
df.to_csv("expense_dataset_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as expense_dataset_cleaned.csv")
