"""Otomatisasi preprocessing dataset Telco Customer Churn.

Konversi dari langkah-langkah manual pada Eksperimen_Iqbal-Rahmatullah.ipynb
(bagian 4. Data Preprocessing) menjadi fungsi yang bisa dijalankan otomatis,
baik lewat CLI maupun diimpor sebagai modul (mis. dari GitHub Actions).
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df, test_size=0.2, random_state=42):
    """Terapkan seluruh langkah preprocessing manual dari notebook eksperimen."""
    df = df.copy()

    # Handling missing value & tipe data
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Drop kolom tidak relevan
    df = df.drop(columns=[ID_COL])

    # Encoding target
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})

    # Encoding fitur kategorikal (one-hot)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Train-test split
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling (fit hanya di train, transform ke train & test)
    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    return train_df, test_df


def save_output(train_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def run(input_path, output_dir, test_size=0.2, random_state=42):
    df = load_data(input_path)
    train_df, test_df = preprocess_data(df, test_size=test_size, random_state=random_state)
    train_path, test_path = save_output(train_df, test_df, output_dir)
    print(f"[automate] train: {train_df.shape} -> {train_path}")
    print(f"[automate] test : {test_df.shape} -> {test_path}")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset Telco Customer Churn")
    parser.add_argument("--input", default="namadataset_raw/telco_customer_churn_raw.csv")
    parser.add_argument("--output", default="preprocessing/telco_customer_churn_preprocessing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
