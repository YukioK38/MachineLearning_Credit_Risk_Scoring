import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path

from preprocess import get_datasets

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["logistic_regression", "random_forest", "XGBoost"]


# Maximum separation between the cdf of the two classes (details on readme)
def compute_ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = max(tpr - fpr)
    
    return(ks)


# Rates the model with the AUC-ROC, KS, precision, recall and f1
def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    result = {
        "model":     name,
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "ks":        round(compute_ks_statistic(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4)
    }

    return(result)



def evaluate_all_models(datasets) -> pd.DataFrame:
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]

    results = []

    for name in MODEL_NAMES:
        print(f"Evaluating {name}")
        model_path = MODELS_DIR / f"{name}.pkl"
        pipeline = joblib.load(model_path)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
    
    results_df = pd.DataFrame(results).set_index("model")
    return(results_df)


# ---------------------------
 
def plot_roc_curves(datasets: dict) -> None:
    """
    Plots ROC curves for all models on the same chart.
    Saves to results/roc_curves.png.
    """
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
 
    plt.figure(figsize=(8, 6))
 
    for name in MODEL_NAMES:
        pipeline = joblib.load(MODELS_DIR / f"{name}.pkl")
        y_prob   = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
 
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Credit Risk Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print("  ROC curves saved to results/roc_curves.png")
 
 
def plot_shap_values(datasets: dict, model_name: str = "xgboost") -> None:
    """
    Generates SHAP summary plot for the specified model.
    Shows which features most influence predictions globally.
    Saves to results/shap_summary.png.
 
    Parameters
    ----------
    model_name : str
        Name of the model to explain (default: xgboost — best performer expected).
    """
    X_test   = datasets["X_test"]
    pipeline = joblib.load(MODELS_DIR / f"{model_name}.pkl")
 
    # Extract the model from the pipeline (SHAP needs the model directly)
    model = pipeline.named_steps["model"]
 
    # TreeExplainer is optimized for tree-based models (RF, XGBoost)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
 
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary saved to results/shap_summary.png")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("Preparing datasets...")
    datasets = get_datasets()
 
    print("\nEvaluating models...")
    results_df = evaluate_all_models(datasets)
 
    print("\nResults:")
    print(results_df.to_string())
 
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    print("\nResults saved to results/model_comparison.csv")
 
    print("\nPlotting ROC curves...")
    plot_roc_curves(datasets)
 
    print("\nGenerating SHAP values for XGBoost...")
    plot_shap_values(datasets)
 
    print("\nEvaluation complete.")
 






