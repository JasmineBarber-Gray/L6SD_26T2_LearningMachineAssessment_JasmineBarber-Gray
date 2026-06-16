import pandas as pd
import numpy as np
import os

# Load data
data = pd.read_excel("data/Net_Worth_Data-1.xlsx")

# Data Analysis
print("FIRST 5 ROWS")
print(data.head())

print("LAST 5 ROWS")
print(data.tail())

print("SHAPE")
print(data.shape)
print(data.shape[0])  # Number of rows
print(data.shape[1])  # Number of columns

print("COLUMN NAMES")
print(data.columns.tolist())

print("MISSING VALUES")
print(data.isnull().sum())

print("DUPLICATES")
print(data.duplicated().sum())

print("DATASET INFO")
data.info()

print("\nSTATISTICAL SUMMARY")
print(data.describe())

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
    "Country",
    "Profession",
    "Healthcare Cost",
    "Education",
    "Gender",
    "Stocks",
    "ETFs",
    "REITs",
    "Bonds"
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