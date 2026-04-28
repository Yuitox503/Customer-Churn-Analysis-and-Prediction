# =====================================
# FEATURE ENGINEERING MODULE
# =====================================

import pandas as pd
from sklearn.model_selection import train_test_split


# =====================================
# LOAD DATA
# =====================================

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# =====================================
# REMOVE LEAKAGE & UNUSED FEATURES
# =====================================

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[
        'Churn Label',
        'Churn Score',
        'Churn Reason'
    ], errors='ignore')

    return df


def drop_high_cardinality_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[
        'City',
        'Zip Code'
    ], errors='ignore')

    return df


# =====================================
# CLEAN CATEGORICAL VALUES
# =====================================

def clean_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })

    return df


# =====================================
# BINARY ENCODING
# =====================================

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    binary_map = {'Yes': 1, 'No': 0}

    for col in df.columns:
        if df[col].dtype == 'object':
            if set(df[col].unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map(binary_map)

    return df


# =====================================
# ONE-HOT ENCODING
# =====================================

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, drop_first=True)
    return df


# =====================================
# FINAL TYPE CLEANING
# =====================================

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(int)
    return df


# =====================================
# SPLIT DATA
# =====================================

def split_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# =====================================
# SAVE DATA
# =====================================

def save_data(X_train, X_test, y_train, y_test, output_dir: str):
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


# =====================================
# MAIN PIPELINE
# =====================================

def run_feature_engineering_pipeline(input_path: str, output_dir: str):
    # Load data
    df = load_data(input_path)

    # Define target
    target = 'Churn Value'

    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    df = drop_high_cardinality_columns(df)

    # Clean categorical values
    df = clean_service_columns(df)

    # Encode features
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)

    # Convert to numeric
    df = convert_types(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target)

    # Save processed data
    save_data(X_train, X_test, y_train, y_test, output_dir)

    return X_train, X_test, y_train, y_test


# =====================================
# EXECUTION
# =====================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_feature_engineering_pipeline(
        input_path="../data/raw/Post_EDA.csv",
        output_dir="../data/processed"
    )

    print("Feature engineering pipeline completed successfully.")