# This file will preproccess the kaggle's Give Me Some Credit dataset

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Directories
RAW_DATA_DIR = Path("data/raw")
KAGGLE = RAW_DATA_DIR/"cs-training.csv"

# We use the standard 80-20 split
TEST_SIZE = 0.2
# For replication
RANDOM_STATE=1


# Dictionary to rename the kaggle columns to be more concise
rename_dict = {
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_days_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}


# load the kaggle dataset from csv
def load_kaggle_data(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    # The first column of the dataset is empty, we can drop it
    df = df.drop(columns=["Unnamed: 0"])
    # rename using the dictionary
    df = df.rename(columns=rename_dict)

    return(df)


# We will replace the missing values with the median and add 
# a dummy variable to indicate wether it was replaced or not
# Also, due to the functionality of the simpleimputer, we must
# create the dummy first, then iput the median
def input_median(df) -> pd.DataFrame:
    df = df.copy()
    inputer = SimpleImputer(strategy="median")
    columns = ["monthly_income", "dependents"]
    df[columns] = inputer.fit_transform(df[columns])

    return(df)

def add_dummy(df) -> pd.DataFrame:
    df = df.copy()
    df["monthly_income_missing"] = df["monthly_income"].isna().astype(int)
    df["dependents_missing"] = df["dependents"].isna().astype(int)

    return(df)


# Separates features (X) from target (y).
def split_feat_target(df):
    X = df.drop(columns=['target'])
    y = df['target']

    return(X, y)


# Full preprocessing pipeline. Loads, cleans, and returns
# train/test splits ready for modeling.
def get_datasets():
    # Load kaggle
    df = load_kaggle_data(KAGGLE)
    
    # Add dummy and median
    df = add_dummy(df)
    df = input_median(df)

    # Separate the features and target
    X, y = split_feat_target(df)

    # Split test and training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Summary:")
    print(f"Training shape: {X_train.shape[0]:,}")
    print(f"Test shape: {X_test.shape[0]:,}")
    print(f"Features shape: {X_train.shape[1]}")
    print(f"Default rate: {y.mean():.2%}")

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    return(result)


if __name__ == "__main__":
    datasets = get_datasets()
    print("Preprocessing complete.")
