# AI-Powered Lead Scoring

Capstone project for the Level 3 Practical AI for Business programme (iMeta / WMCA, 2025).

Sales teams waste time on leads that will never close. This project trains a model on historical leads, scores new ones with a probability of winning, and splits them into Hot / Warm / Cold so outreach can be prioritised.

## What it does
- **`train_evaluate.py`** – command-line pipeline: loads historical leads, builds a scikit-learn preprocessing pipeline (imputation, scaling, one-hot encoding), compares Logistic Regression and Random Forest, reports ROC-AUC and PR-AUC, saves the best model.
- **`app.py`** – Streamlit dashboard with two tabs: *Train* (upload history, retrain, download model) and *Score* (upload model + new leads, download scored file).
- **`segmentation.py`** – splits scored leads into Hot / Warm / Cold thirds.

## Run it
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train_evaluate.py --in leads_history.csv --out lead_scoring_model.pkl
streamlit run app.py
```

## A note on the results
The sample data (`leads_history.csv`, `new_leads.csv`) is **synthetic**, generated for the exercise, so it carries no real buying signal. The validation ROC-AUC is about 0.41 (no better than chance), which is the honest outcome on random data. The project demonstrates the pipeline — preprocessing, model comparison, evaluation and a usable interface — not a production model. Next step: train on real CRM opportunity history with features such as stage velocity and engagement recency.

## Stack
Python · pandas · scikit-learn · Streamlit · matplotlib · joblib
