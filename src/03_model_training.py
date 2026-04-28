# =====================================
# MODEL TRAINING MODULE
# =====================================

import pandas as pd
import pickle

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Evaluation
from sklearn.metrics import accuracy_score, classification_report


# =====================================
# LOAD DATA
# =====================================

def load_data():
    X_train = pd.read_csv("../data/processed/X_train.csv")
    X_test = pd.read_csv("../data/processed/X_test.csv")

    y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


# =====================================
# SCALE DATA (FOR LOGISTIC REGRESSION)
# =====================================

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# =====================================
# TRAIN BASE MODELS
# =====================================

def train_logistic_regression(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# =====================================
# TRAIN OPTIMIZED MODELS
# =====================================

def train_optimized_logistic(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    return model


def train_optimized_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# =====================================
# EVALUATION FUNCTION
# =====================================

def evaluate_model(model, X_test, y_test, threshold=None):
    if threshold:
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# =====================================
# FEATURE IMPORTANCE (LOGISTIC)
# =====================================

def get_feature_importance(model, feature_names):
    importances = model.coef_[0]

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    feat_df['Abs Importance'] = feat_df['Importance'].abs()
    feat_df = feat_df.sort_values(by='Abs Importance', ascending=False)

    return feat_df


# =====================================
# SAVE MODELS
# =====================================

def save_models(models: dict):
    for name, model in models.items():
        with open(f"../models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)


# =====================================
# MAIN PIPELINE
# =====================================

def run_training_pipeline():

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Scale data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Train models
    log_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train, y_train)

    log_model_opt = train_optimized_logistic(X_train_scaled, y_train)
    rf_model_opt = train_optimized_rf(X_train, y_train)

    print("\n--- Logistic Regression ---")
    evaluate_model(log_model, X_test_scaled, y_test)

    print("\n--- Random Forest ---")
    evaluate_model(rf_model, X_test, y_test)

    print("\n--- Logistic Regression (Optimized) ---")
    evaluate_model(log_model_opt, X_test_scaled, y_test, threshold=0.3)

    print("\n--- Random Forest (Optimized) ---")
    evaluate_model(rf_model_opt, X_test, y_test)

    # Feature importance
    feat_df = get_feature_importance(log_model_opt, X_train.columns)
    print("\nTop Features:\n", feat_df.head(10))

    # Save models
    save_models({
        "log_model": log_model,
        "rf_model": rf_model,
        "log_model_optimized": log_model_opt,
        "rf_model_optimized": rf_model_opt,
        "scaler": scaler
    })

    return feat_df


# =====================================
# EXECUTION
# =====================================

if __name__ == "__main__":
    run_training_pipeline()
    print("Model training pipeline completed successfully.")