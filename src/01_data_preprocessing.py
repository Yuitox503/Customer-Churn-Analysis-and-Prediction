# =====================================
# DATA PREPROCESSING MODULE
# =====================================

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    df = df.drop(
        columns=['CustomerID', 'Count', 'Lat Long', 'Latitude', 'Longitude'],
        errors='ignore'
    )

    # Convert Total Charges to numeric
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # Fix Zip Code datatype
    df['Zip Code'] = df['Zip Code'].astype('object')

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    # Fill missing Total Charges with 0
    df['Total Charges'] = df['Total Charges'].fillna(0)

    return df


def get_churn_distribution(df: pd.DataFrame) -> pd.Series:
    return df['Churn Label'].value_counts(normalize=True)


def get_churn_by_contract(df: pd.DataFrame) -> pd.Series:
    return df.groupby('Contract')['Churn Value'].mean().sort_values(ascending=False)


def get_churn_by_internet(df: pd.DataFrame) -> pd.Series:
    return df.groupby('Internet Service')['Churn Value'].mean().sort_values(ascending=False)


def save_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)


# =====================================
# MAIN PIPELINE
# =====================================

def run_pipeline(input_path: str, output_path: str):

    # Load data
    df = load_data(input_path)

    # Clean data
    df = clean_data(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Save processed data
    save_data(df, output_path)

    return df


# =====================================
# EXECUTION
# =====================================

if __name__ == "__main__":
    df_processed = run_pipeline(
        input_path="../data/raw/Customer_Churn_Raw.xlsx",
        output_path="../data/processed/Post_EDA.csv"
    )

    print("Data preprocessing completed successfully.")