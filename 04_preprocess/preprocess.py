import pandas as pd
import numpy as np

INPUT_FILE = r"E:\codes\imgadv\03_getprice\output\all_native_prices_2025.csv"
OUTPUT_FILE = r"E:\codes\imgadv\04_preprocess\outputs\data.csv"

def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Apply min-max normalization to each column (except 'date')
    norm_df = df.copy()
    for col in norm_df.columns:
        if col == 'date':
            continue
        col_data = pd.to_numeric(norm_df[col], errors='coerce')
        min_val = col_data.min(skipna=True)
        max_val = col_data.max(skipna=True)
        # Avoid division by zero
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            norm_df[col] = np.nan
        else:
            norm_df[col] = (col_data - min_val) / (max_val - min_val)
    return norm_df

def print_data_shape(df: pd.DataFrame):
    n_samples, n_features = df.shape
    # Exclude 'date' from feature count if present
    feature_cols = [c for c in df.columns if c != 'date']
    print(f"データ次元: サンプル数={n_samples}, 特徴量数={len(feature_cols)}")

def main():
    # Read CSV
    df = pd.read_csv(INPUT_FILE)
    # Show data shape
    print_data_shape(df)
    # Normalize
    norm_df = minmax_normalize(df)
    # Save to CSV
    norm_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()