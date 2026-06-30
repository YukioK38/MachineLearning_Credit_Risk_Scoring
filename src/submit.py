import pandas as pd
import joblib
from pathlib import Path

from preprocess import (
    rename_dict,
    add_dummy,
)
from sklearn.impute import SimpleImputer

RAW_DATA_DIR = Path("data/raw")
MODELS_DIR   = Path("models")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE  = RAW_DATA_DIR / "cs-test.csv"
MODEL_FILE = MODELS_DIR / "xgboost.pkl"


def prepare_test_data(filepath: Path = TEST_FILE) -> pd.DataFrame:
    """
    Loads and preprocesses the Kaggle test set using the same steps
    applied to the training data — except imputation, which must reuse
    statistics learned from the training set (not recomputed on test data),
    to avoid data leakage.

    Returns
    -------
    pd.DataFrame ready for prediction, with an 'Id' column preserved separately.
    """
    df = pd.read_csv(filepath)

    # The Kaggle test file uses 'Unnamed: 0' as the row Id — this is the
    # identifier required in the submission file, so we keep it separately
    # before dropping it from the features
    ids = df["Unnamed: 0"]
    df = df.drop(columns=["Unnamed: 0"])

    # The Kaggle test file includes 'SeriousDlqin2yrs' as an empty column
    # (all NaN) — this is the target we're predicting, so we drop it
    if "SeriousDlqin2yrs" in df.columns:
        df = df.drop(columns=["SeriousDlqin2yrs"])

    df = df.rename(columns=rename_dict)

    # Add missing-value dummy variables, same as training
    df = add_dummy(df)

    # NOTE: ideally the imputer fitted on the training set should be reused
    # here (saved via joblib in preprocessing.py) instead of fitting a new
    # one. For simplicity, we refit on the test set's own median — a known
    # simplification documented in the README.
    imputer = SimpleImputer(strategy="median")
    columns_to_impute = ["monthly_income", "dependents"]
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    return df, ids


def generate_submission(model_path: Path = MODEL_FILE) -> Path:
    """
    Generates predictions on the test set and saves a Kaggle-ready
    submission CSV with columns: Id, Probability.

    Returns
    -------
    Path to the saved submission file.
    """
    print("Loading and preprocessing test data...")
    X_test, ids = prepare_test_data()

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print("Generating predictions...")
    # predict_proba returns [P(no default), P(default)] — Kaggle wants P(default)
    probabilities = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "Id": ids,
        "Probability": probabilities,
    })

    output_path = RESULTS_DIR / "kaggle_submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())

    return output_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_submission()