import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load cleaned data
data = pd.read_csv("data/net_worth_data_cleaned.csv")

# Features and target
features = [
    "Age",
    "Income",
    "Credit Card Debt",
    "Stocks",
    "Mutual Funds"
]

X = data[features]
y = data["Net Worth"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42)
}

results = {}

# Train and evaluate each model
for name, model in models.items():

    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    MAE = mean_absolute_error(y_test, preds)
    RMSE = np.sqrt(mean_squared_error(y_test, preds))
    R2 = r2_score(y_test, preds)

    results[name] = {
        "MAE": MAE,
        "RMSE": RMSE,
        "R2": R2,
        "model": model
    }

    print(f"\n{name}")
    print(f"MAE: {MAE:,.2f}")
    print(f"RMSE: {RMSE:,.2f}")
    print(f"R2: {R2:.4f}")

# Best model selection
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = results[best_model_name]["model"]

print("\nBEST MODEL:", best_model_name)

# Results table
results_table = pd.DataFrame({
    name: {
        "MAE": results[name]["MAE"],
        "RMSE": results[name]["RMSE"],
        "R2": results[name]["R2"]
    }
    for name in results
}).T

print("\nMODEL COMPARISON:")
print(results_table)

# Folders
os.makedirs("model", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# Graph 1 - R² Comparison
plt.figure()
plt.bar(results.keys(), [results[m]["R2"] for m in results])
plt.title("R² Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Graph 2 - RMSE
plt.figure()
plt.bar(results.keys(), [results[m]["RMSE"] for m in results])
plt.title("RMSE Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Graph 3 - Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data[features + ["Net Worth"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (5 Features)")
plt.tight_layout()
plt.show()

# Graph 4 - Error Distribution
preds = best_model.predict(X_test_scaled if best_model_name == "Linear Regression" else X_test)
errors = y_test - preds

plt.figure()
plt.hist(errors, bins=25, edgecolor="black")
plt.axvline(0, color="red")
plt.title("Error Distribution")
plt.tight_layout()
plt.show()

# Graph 5 - Actual vs Predicted
plt.figure()
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "g--")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()

# Save model + scaler
joblib.dump(best_model, "model/networth_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nSaved model + scaler successfully!")