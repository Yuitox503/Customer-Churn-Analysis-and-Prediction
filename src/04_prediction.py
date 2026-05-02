# =====================================
# PREDICTION PIPELINE 
# =====================================

import pandas as pd
import pickle


# =====================================
# LOAD MODEL
# =====================================

def load_model():

    with open("models/rf_model_optimized.pkl", "rb") as f:
        model = pickle.load(f)

    return model


# =====================================
# PREPROCESSING FUNCTIONS
# =====================================

def clean_data(df):
    df = df.drop(
        columns=['CustomerID', 'Count', 'Lat Long', 'Latitude', 'Longitude'],
        errors='ignore'
    )

    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Zip Code'] = df['Zip Code'].astype('object')

    return df


def handle_missing_values(df):
    df['Total Charges'] = df['Total Charges'].fillna(0)
    return df


def drop_unnecessary_columns(df):
    df = df.drop(columns=[
        'Churn Label',
        'Churn Score',
        'Churn Reason'
    ], errors='ignore')

    return df


def drop_high_cardinality_columns(df):
    df = df.drop(columns=[
        'City',
        'Zip Code'
    ], errors='ignore')

    return df


def clean_service_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })
    return df


def encode_binary_columns(df):
    binary_map = {'Yes': 1, 'No': 0}

    for col in df.columns:
        if df[col].dtype == 'object':
            if set(df[col].unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map(binary_map)

    return df


def encode_categorical_columns(df):
    df = pd.get_dummies(df, drop_first=True)
    return df


def convert_types(df):
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    return df


# =====================================
# ALIGN COLUMNS
# =====================================

def align_columns(df, model_features):

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]
    return df


# =====================================
# MAIN PIPELINE
# =====================================

def run_prediction_pipeline():

    # Load raw data
    df_raw = pd.read_excel("data/raw/Customer_Churn_Raw.xlsx")

    # Load model
    model = load_model()

    # Apply preprocessing
    df = clean_data(df_raw.copy())
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df = drop_high_cardinality_columns(df)
    df = clean_service_columns(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = convert_types(df)

    # Remove target
    df = df.drop(columns=['Churn Value'], errors='ignore')

    # Align columns
    model_features = model.feature_names_in_
    df = align_columns(df, model_features)

    # Predict probabilities
    probs = model.predict_proba(df)[:, 1]

    # Attach results to original dataset
    df_raw['churn_probability'] = probs
    df_raw['churn_risk'] = (probs > 0.4).astype(int)

    # Save
    df_raw.to_csv("data/processed/predictions.csv", index=False)

    print("Predictions generated successfully.")


# =====================================
# EXECUTION
# =====================================

if __name__ == "__main__":
    run_prediction_pipeline()