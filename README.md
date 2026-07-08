# Machine Learning Credit Risk Scoring

## Try the [Streamlit App](https://dashboard-simulation.streamlit.app/)!

A **credit risk scoring** model comparing Logistic Regression, Random Forest, and XGBoost, enriched with macroeconomic data collected via the FRED (Federal Reserve) API, and interpreted with SHAP to address requirements under **LGPD (Brazil's data protection law), Art. 20**.

> **TL;DR:** The best model (XGBoost) achieved an **AUC-ROC of 0.8603**, corresponding to a public score of **0.86204** on Kaggle's *Give Me Some Credit* competition, good enough for the **TOP 100** (1st place: 0.869).

## Table of Contents

- [About the project](#-about-the-project)
- [Repository structure](#-repository-structure)
- [Tech stack](#-tech-stack)
- [Getting started](#-getting-started)
- [Methodology](#-methodology)
- [Results](#-results)
- [Interpretability (SHAP) and LGPD](#-interpretability-shap-and-lgpd)
- [Full article](#-full-article)
- [Contact](#-contact)
- [License](#-license)

## About the project

The goal is to predict the probability that a customer will default, using data from Kaggle's [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview) competition. The project covers the full pipeline:

- Collecting macroeconomic data via API (FRED, Federal Reserve Economic Data);
- Handling missing data and class imbalance;
- Training and comparing three models (Logistic Regression, Random Forest, XGBoost);
- Evaluating results with multiple metrics (AUC-ROC, KS statistic, Precision, Recall, F1);
- Interpreting the final model with SHAP, in line with LGPD Art. 20;
- Submitting and validating the result on Kaggle.

## Repository structure

MachineLearning_Credit_Risk_Scoring/

├─ data/                  # Raw and processed data (cs-training.csv, cs-test.csv)

├─ notebooks/             # Exploratory analysis and plots

├─ src/                   # Source code (FRED collection, pipeline, training, evaluation)

├─ models/                # Trained models (.pkl)

├─ outputs/                # Results, plots, and Kaggle submission

├─ requirements.txt

└─ README.md

##  Tech stack

- Python 3.x
- pandas / numpy
- scikit-learn (Pipeline, StandardScaler, LogisticRegression, RandomForestClassifier)
- XGBoost
- SHAP
- requests (FRED API consumption)
- joblib (model persistence)

## Getting started

```bash
# install dependencies
pip install -r requirements.txt
```

### FRED API key

The project uses the [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html) to collect macroeconomic time series. Create a free account, get your key, and set it (for example, as an environment variable):

```bash
export FRED_API_KEY="your_key_here"
```

### Running the pipeline

```bash
# Train models
python src/train.py
# Evaluate models
python src/evaluate.py
# Create a submission
python src/submit.py
# Run the streamlit app
streamlit run app/app.py
```

## Methodology

1. **Missing data handling:** median imputation (fitted on the training set only, to avoid data leakage) plus dummy indicator variables flagging missing information.
2. **Class imbalance:** only ~6.7% of the dataset is delinquent. Addressed with `class_weight="balanced"` (Logistic Regression and Random Forest) and `scale_pos_weight` (XGBoost).
3. **Pipeline (Scikit-learn):** standardization (`StandardScaler`) + model, ensuring `fit_transform` is applied only on training data and `transform` on test data.
4. **Macroeconomic validation (Appendix):** FRED series couldn't be used as model features (the Kaggle dataset doesn't include the loan date), but they were used to externally validate the temporal plausibility of the dataset, showing consistency with the 2008–2010 financial crisis.

Mathematical details for each model (Logistic Regression, Random Forest, and XGBoost) are available in the full article's Appendix section.

## Results

| Model | AUC-ROC | KS | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7849 | 0.4221 | 0.1696 | 0.6384 | 0.2680 |
| Random Forest | 0.8559 | 0.5611 | 0.2290 | 0.7257 | 0.3481 |
| **XGBoost** | **0.8603** | **0.5647** | 0.2141 | 0.7631 | 0.3344 |

**XGBoost** was selected as the final model for outperforming the others on three of the five metrics, particularly AUC-ROC, the metric used to rank submissions on Kaggle.

**Kaggle submission:** public score of **0.86204**, a difference of only ~0.0017 from the score obtained on the local test sample, indicating good generalization (no strong evidence of overfitting).

## Interpretability (SHAP) and LGPD

Article 20 of Brazil's [LGPD (Law No. 13,709/2018)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm) grants data subjects the right to request clear information about the criteria used in automated credit decisions. For this reason, the final model (XGBoost) was analyzed with **SHAP**, making it possible to identify each feature's contribution to the individual decision for every customer.

The three features with the highest mean absolute SHAP value were:

1. `open_credit_lines`: number of open credit lines;
2. `past_due_30_59`: moderate delinquency (30 to 59 days past due);
3. `times_90_days_late`: severe delinquency (more than 90 days past due).

## Full article

A detailed walkthrough of every decision and trade-off in this project is available in the article: *"Machine Learning Credit Risk Scoring"* (included in this repository / linked on the author's profile).

## Contact

Open to Data Science opportunities:

- LinkedIn: [fabio-kitsuwa](https://br.linkedin.com/in/fabio-kitsuwa)
- GitHub: [YukioK38](https://github.com/YukioK38)

## License

Code: licensed under the MIT License.
Article: licensed under CC BY 4.0.
