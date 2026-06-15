import pandas as pd
import numpy as np
import os

# Load data
DATA_PATH = "data/Net_Worth_Data-1.xlsx"


def load_data():
    return pd.read_excel(DATA_PATH)


def clean_data(data):
    data = data.dropna()
    data = data.drop_duplicates()

    data = data.drop(columns=[
        "Client Name",
        "Client e-mail",
        "Country"
    ], errors="ignore")

    return data

# Data Analysis
def analyze(data):
    return {
        "shape": data.shape,
        "missing": data.isnull().sum().sum(),
        "duplicates": data.duplicated().sum(),
        "columns": data.columns.tolist(),
        "summary": data.describe()
    }
    
if __name__ == "__main__":
    data = load_data()
    print(data.head())

# Cleaning Data
before_rows = data.shape[0]

data = data.dropna()  # Remove rows with missing values
data = data.drop_duplicates()  # Remove duplicate rows

after_rows = data.shape[0]

print(f"\nRows removed during cleaning: {before_rows - after_rows}")

# Privacy and Ethical Handling of Data
columns_to_drop = [
    "Client Name",
    "Client e-mail",
    "Country"
]

data_cleaned = data.drop(columns=columns_to_drop, errors="ignore")

print("\nColumns Removed:")
print(columns_to_drop)

print("\nRemaining Columns:")
print(data_cleaned.columns.tolist())

# Save cleaned data for model training

os.makedirs("data", exist_ok=True)

data_cleaned.to_csv("data/net_worth_data_cleaned.csv", index=False)

print("\nCleaned data saved to 'data/net_worth_data_cleaned.csv'")

print("\nCORRELATION WITH NET WORTH")
print(
    data_cleaned.corr(numeric_only=True)["Net Worth"]
    .sort_values(ascending=False)
)

print("\nNET WORTH SUMMARY")
print(data_cleaned["Net Worth"].describe())