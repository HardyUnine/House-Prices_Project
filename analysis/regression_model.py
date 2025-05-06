import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/clean_train.csv")

# Drop rows with missing SalePrice (if any)
df = df.dropna(subset=["SalePrice"])

# Separate target
y = df["SalePrice"]

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("SalePrice")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

# Combine all features
X = pd.concat([df[numeric_cols], df_encoded], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant term for intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit OLS regression
model = sm.OLS(y_train, X_train_const).fit()

# Print model summary
print(model.summary())

# Predict and evaluate
y_pred = model.predict(X_test_const)
r2 = r2_score(y_test, y_pred)
print(f"R² on test set: {r2:.4f}")

# Optional: plot predicted vs actual
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/regression_pred_vs_actual.png")
plt.show()
