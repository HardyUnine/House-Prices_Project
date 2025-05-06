import pandas as pd
import numpy as np
import os

# Path to your dataset
DATA_PATH_RAW = '/Users/keenanhardy/Desktop/SCD 4e Semestre/CSTATS/House-Prices_Project/data/raw/'
DATA_PATH_CLEAN = '/Users/keenanhardy/Desktop/SCD 4e Semestre/CSTATS/House-Prices_Project/data/clean'


INPUT_FILE = os.path.join(DATA_PATH_RAW, 'train.csv')
OUTPUT_FILE = os.path.join(DATA_PATH_CLEAN, 'clean_train.csv')

# Load the raw dataset
df = pd.read_csv(INPUT_FILE)

# 1. Drop columns with more than 30% missing values
threshold = 0.3
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > threshold].index
df = df.drop(columns=cols_to_drop)

# 2. Separate by type
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# 3. Fill missing numeric values with median
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# 4. Fill missing categorical values with mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# 5. Save cleaned data
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Cleaned data saved to: {OUTPUT_FILE}")
