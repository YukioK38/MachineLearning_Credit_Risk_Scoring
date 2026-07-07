import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🔎",
    layout="wide"
)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

@st.cache_resource
def load_model(model_name):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    return joblib.load(model_path)

@st.cache_data
def load_comparison_results() -> pd.DataFrame:
    return(pd.read_csv(RESULTS_DIR / "model_comparison.csv", index_col="model"))

st. title("Credit Risk Scoring")
st.caption("Predicting the probability of serious deliquency within 2 years")

tab1, tab2 = st.tabs(["Model dashboard", "Individual Score"])

with tab1:
    st.header("Model Performance Dashboard")

    st.subheader("Model Comparison")
    
    results_df = load_comparison_results()
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="lightgreen"),
        use_container_width=True
    )
    st.caption(
        "XGBoost and Random Forest substantially outpeform Logistic Regression"
        "on KS Statistic and AUC-ROC. The gap between XGBoost and Random Forest"
        "is small (~0.005 AUC)"
    )

    st.divider()

    st.subheader("ROC Curves")

    roc_path = RESULTS_DIR / "roc_curves.png"
    st.image(str(roc_path), use_container_width=True)

    st.divider()

    st.subheader("Feature Importantece (SHAP - XGBoost)")

    shap_path = RESULTS_DIR / "shap_summary.png"
    st.image(str(shap_path), use_container_width=True)
    st.caption("Each point is a client. Red = high feature value, blue = low"
               "Position on the x_axis shows the impact on the prediction")

    st.divider()

    st.subheader("Macroeconomic Context (FRED)")
    fred_path = RESULTS_DIR / "fred_vs_kaggle.png"
    st.image(str(fred_path), use_container_width=True)
    st.caption(
        "The Kaggle dataset's aggregate default rate (6.68%) most closely"
        "matches the FRED deliquency around April 2009, remaining inside the" \
        "financial crisis perios. This suggests the dataset's features are already" \
        "capturing the macroeconimic state of the period, so we won't be" \
        "adding the FRED variables as features to avoid colinearity (see README)"
    )

    st.divider()

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total clients", "150,000")
    col2.metric("Default Rate", "6.68%")
    col3.metric("Features Used", 12)
    

with tab2:
    st.header("Predict Risk for a single client")
    model_choice = st.selectbox(
        "Select a model",
    options=["XGBoost", "random_forest", "logistic_regression"]
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1: 
        age = st.number_input("Age", min_value=18, max_value=100)
        monthly_income = st.number_input("Monthly income (USD $)", min_value=0, step=100)
        dependents = st.number_input("Number of dependets", min_value=0, max_value=20)

    with col2:
        debt_ratio = st.slider(
            "Debt Ratio", min_value=0.0, max_value=2.0, step=0.01,
            help="Monthly debt payments divided by monthly income"
        )
        revolving_utilization = st.slider(
            "Revolving Credit Utilization", min_value=0.0, max_value=2.0, step=0.01,
            help="Percentage of available revolving credit currently in use"
        )
        open_credit_lines = st.number_input(
            "Open Credit Lines and Loans", min_value=0, max_value=50
        )

    with col3:
        past_due_30_59 = st.number_input(
            "Times 30-59 Days Past Due", min_value=0, max_value=20
        )
        past_due_60_89 = st.number_input(
            "Times 60-89 Days Past Due", min_value=0, max_value=20
        )
        times_90_days_late = st.number_input(
            "Times 90+ Days Past Due", min_value=0, max_value=20
        )
        real_estate_loans = st.number_input(
            "Real estate loans or lines", min_value=0, max_value=20
        )

    st.divider()

    if st.button("Predict Risk", type="primary"):
        input_data = pd.DataFrame([{
                "revolving_utilization":   revolving_utilization,
                "age":                     age,
                "past_due_30_59":          past_due_30_59,
                "debt_ratio":              debt_ratio,
                "monthly_income":          monthly_income,
                "open_credit_lines":       open_credit_lines,
                "times_90_days_late":      times_90_days_late,
                "real_estate_loans":       real_estate_loans,
                "past_due_60_89":          past_due_60_89,
                "dependents":              dependents,
                "monthly_income_missing":  0,  # user always provides this in the form
                "dependents_missing":      0,  # same here
        }])

        model = load_model(model_choice)

        risk_probability = float(model.predict_proba(input_data)[0][1])

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(
                label="Default Risk",
                value = f"{risk_probability:.1%}"
            )

        with result_col2:
            if risk_probability <0.10:
                st.success("Low risk: likely to be approved")
            elif risk_probability < 0.3:
                st.warning("Moderate risk: may require additional review")
            else: st.error("High risk: likely declined")

        st.progress(min(risk_probability, 1.0))

