import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ==========================================
# LAB1: Dataset Exploration
# ==========================================
print("=== LAB1: Dataset Exploration ===")

# 1. Load Dataset
df = pd.read_csv('amazon.csv') 
print("\n[1] First 2 rows of the dataset:")
print(df.head(2))

# 2. Display Shape
print("\n[2] Dataset Dimensions (Rows, Columns):")
print(df.shape)


# --- DATA CLEANING & TYPE CONVERSION ---
# Remove special characters to convert string values into numerical types
columns_to_clean = ['discounted_price', 'actual_price']
for col in columns_to_clean:
    if col in df.columns:
        # Strip currency symbol and commas
        df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'discount_percentage' in df.columns:
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '')
    df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')

if 'rating' in df.columns:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

if 'rating_count' in df.columns:
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
# --------------------------------------------------------


# 3. Display Data Types
print("\n[3] Data types of each column (Post-Conversion):")
print(df.dtypes)

# 4. Display Summary Statistics
print("\n[4] Summary Statistics:")
print(df.describe()) 

# 5. Display Missing Values
print("\n[5] Number of Missing Values per column:")
print(df.isnull().sum())

# 6. Display Duplicate Records
print("\n[6] Check for Duplicate Records:")
print("Total number of duplicate rows:", df.duplicated().sum())

# 7. Display Class Distribution
if 'rating' in df.columns:
    print("\n[7] Data distribution in the 'rating' column:")
    print(df['rating'].value_counts().head())


# ==========================================
# LAB2: Data Visualization
# ==========================================
print("\n=== LAB2: Data Visualization ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns

if len(numeric_cols) > 0:
    target_col = 'rating' if 'rating' in numeric_cols else numeric_cols[0]
    
    # 1. Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col].dropna(), kde=True, color='skyblue')
    plt.title(f'Histogram of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.show()

    # 2. Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()
else:
    print("No numerical columns found for data visualization.")


# ==========================================
# Part 3: Data Cleaning
# ==========================================
print("\n=== Part 3: Data Cleaning ===")
df_clean = df.copy()

# 1. Missing Value Handling & Compare Mean vs Median
if 'rating' in df_clean.columns:
    fill_col = 'rating'
    mean_value = df_clean[fill_col].mean()
    median_value = df_clean[fill_col].median()
    
    print(f"Column '{fill_col}' -> Mean: {mean_value:.2f} | Median: {median_value:.2f}")
    
    # Impute missing values using the median value
    df_clean[fill_col] = df_clean[fill_col].fillna(median_value)
    print(f"Missing values in '{fill_col}' have been successfully imputed with Median.")

# 2. Duplicate Removal
initial_rows = df_clean.shape[0]
df_clean.drop_duplicates(inplace=True)
print(f"Removed duplicate rows: Reduced from {initial_rows} rows to {df_clean.shape[0]} rows.")

# 3. Incorrect Data Correction & Data Type Conversion
# Impute missing values in the remaining numerical columns with their respective median values
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())


# ==========================================
# Part 4: Feature Engineering
# ==========================================
print("\n=== Part 4: Feature Engineering ===")
categorical_cols = df_clean.select_dtypes(include=['object']).columns

if len(categorical_cols) > 0:
    encode_col = 'category' if 'category' in categorical_cols else categorical_cols[0]
    print(f"Selected column '{encode_col}' for Categorical Encoding.")
    
    # Ensure all missing or text entries are processed as string types
    df_clean[encode_col] = df_clean[encode_col].astype(str)
    
    # 1. Label Encoding
    le = LabelEncoder()
    df_clean[f'{encode_col}_LabelEnc'] = le.fit_transform(df_clean[encode_col])
    print("- Label Encoding completed (New column created).")
    
    # 2. One-Hot Encoding (Testing on the top 5 categories to manage computational complexity)
    top_categories = df_clean[encode_col].value_counts().head(5).index
    df_subset = df_clean[df_clean[encode_col].isin(top_categories)]
    df_onehot = pd.get_dummies(df_subset, columns=[encode_col], drop_first=True)
    print("- One-Hot Encoding (Top 5 Categories) completed.")
    
    print("\n=== Preprocessing Script Finished Successfully ===")
else:
    print("No categorical columns found for Feature Engineering.")