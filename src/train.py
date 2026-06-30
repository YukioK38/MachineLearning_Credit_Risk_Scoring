import pandas as pd
import joblib
from pathlib import Path
 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
 
from preprocess import get_datasets

MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
RANDOM_STATE = 1

# Define the models that are going to be used:
# It is important to consider the class_weight and the scale_pos_weight
# for the XGBoost, since our sample isn't balanced
MODELS = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    # Note on memory: The computer used to make this project
    # has 8 Gb of RAM, so we must limit the depth of the random
    # forest
    "random_forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=RANDOM_STATE
    ),
    # In order to mantain compatibility with the pipeline and
    # control the overfitting
    "XGBoost": XGBClassifier(
        scale_pos_weight=13,
        n_estimators=100,
        max_depth=4,          
        learning_rate=0.1, 
        subsample=0.8, # use 80% of the sample per tree
        colsample_bytree=0.8, # use 80% of features by tree
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0,
    )
}

# builds a pipeline for the model argument
def build_pipeline(model) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return(pipeline)


# Train and save the results as a pkl, also returns it in ram
def train_models(datasets) -> dict:
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]

    result_models = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in MODELS.items():
        print(f"Training {name}")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(pipeline, model_path)
        print(f"Saved to {model_path}")

        result_models[name] = pipeline

    return(result_models)

if __name__ == "__main__":
    datasets = get_datasets()

    trained_models = train_models(datasets)

    print("Training complete")


